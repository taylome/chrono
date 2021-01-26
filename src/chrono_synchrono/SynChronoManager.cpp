#include "chrono_synchrono/SynChronoManager.h"

#include "chrono_synchrono/SynConfig.h"
#include "chrono_synchrono/utils/SynLog.h"
#include "chrono_synchrono/agent/SynAgentFactory.h"

#ifdef CHRONO_FASTDDS
#undef ALIVE

#include "chrono_synchrono/communication/dds/SynDDSCommunicator.h"
#include "chrono_synchrono/communication/dds/SynDDSListener.h"
#include "chrono_synchrono/communication/dds/idl/SynDDSMessage.h"
#include "chrono_synchrono/communication/dds/idl/SynDDSMessagePubSubTypes.h"
#include "chrono_synchrono/communication/dds/SynDDSTopic.h"
#endif

namespace chrono {
namespace synchrono {

#ifdef CHRONO_FASTDDS

void ProcessMessage(std::shared_ptr<SynCommunicator> communicator, void* message) {
    communicator->ProcessBuffer(((SynDDSMessage*)message)->data());
}

void RegisterParticipant(std::shared_ptr<SynCommunicator> communicator, const std::string& participant_name) {
    std::string default_prefix = "/syn/node/";

    std::size_t found = participant_name.find(default_prefix);
    if (found == std::string::npos)
        return;  // prefix not found

    if (auto dds_communicator = std::dynamic_pointer_cast<SynDDSCommunicator>(communicator)) {
        SynLog() << "Adding Participant: " << participant_name << "\n";

        auto callback = std::bind(&ProcessMessage, communicator, std::placeholders::_1);
        dds_communicator->CreateSubscriber(SynDDSTopic::RemovePrefix(participant_name), new SynDDSMessagePubSubType(),
                                           callback, new SynDDSMessage(), true);
    }
}

#endif

SynChronoManager::SynChronoManager(SynNodeID nid, SynAgentNum num_nodes, std::shared_ptr<SynCommunicator> communicator)
    : m_nid(nid), m_num_nodes(num_nodes), m_initialized(false), m_heartbeat(1e-2), m_next_sync(0.0) {
    if (communicator)
        SetCommunicator(communicator);

    // Set logger for logging prefix
    SetLogNodeID(nid);
}

SynChronoManager::~SynChronoManager() {}

/// Unique id is a single integer where the last digit is the order of which it is added to the node
/// TODO: Right now, a node can have a maximum of 10 agents (0 through 9 digits)
// Set the agent at the specified node
bool SynChronoManager::AddAgent(std::shared_ptr<SynAgent> agent) {
    // Because it is assumed a handshake is done when the Initialization function is called,
    // it is not allowed to add a new agent after this process to ensure each node/agent knows
    // about each other
    if (m_initialized) {
        SynLog() << "WARNING: SynChronoManager has been initialized. All agents should be added prior to "
                    "initializing the manager. Ignoring this new agent.\n";
        return false;
    }

    // Get the most recently added agent on this node and increment it's num by one.
    // If no agents have been added, the agent's id will default to 0
    SynAgentNum new_agent_num = m_agents.empty() ? 0 : m_agents.rbegin()->first + 1;

    // concatenate the node id and the agent id
    SynAgentID aid = std::stoul(std::to_string(m_nid) + std::to_string(new_agent_num));
    m_agents[aid] = agent;

    agent->SetID(aid);

#ifdef CHRONO_FASTDDS
    if (auto dds_communicator = std::dynamic_pointer_cast<SynDDSCommunicator>(m_communicator)) {
        // Create the topic that state information will be passed on
        // and add the topic to the communicator
        auto topic_name = "node/" + std::to_string(aid);
        dds_communicator->CreatePublisher(topic_name, new SynDDSMessagePubSubType());
    }
#endif

    return true;
}

bool SynChronoManager::AddZombie(std::shared_ptr<SynAgent> zombie, SynAgentID aid) {
    // Because it is assumed a handshake is done when the Initialization function is called,
    // it is not allowed to add a new zombie after this process to ensure each node/agent knows
    // about each other
    if (m_initialized) {
        SynLog() << "WARNING: SynChronoManager has been initialized. All zombies should be added prior to "
                    "initializing the manager. Ignoring this new zombie.\n";
        return false;
    }

    m_zombies[aid] = zombie;
    zombie->SetID(aid);

    return true;
}

bool SynChronoManager::SetCommunicator(std::shared_ptr<SynCommunicator> communicator) {
    // Because it is assumed a handshake is done when the Initialization function is called,
    // it is not allowed to set the communicator after this process to ensure each node/agent knows
    // about each other
    if (m_initialized) {
        SynLog() << "WARNING: SynChronoManager has been initialized. The communicator should be set prior to "
                    "initializing the manager. Ignoring the communicator.\n";
        return false;
    }

    m_communicator = communicator;

    return true;
}

bool SynChronoManager::Initialize(ChSystem* system) {
    if (!m_communicator) {
        SynLog() << "WARNING: A Communicator has not been attached.\n";
        return false;
    }

    // Initialize the communicator
    m_communicator->Initialize();

#ifdef CHRONO_FASTDDS
    // If the communicator uses DDS, we want to create subscribers that will listen to state information
    // coming from the other nodes. This is done by setting the name of each governing participant to
    // common names to be parsed. RegisterParticipant will parse these names and create Subscribers
    // listening to incoming state data.
    if (auto dds_communicator = std::dynamic_pointer_cast<SynDDSCommunicator>(m_communicator)) {
        dds_communicator->Barrier(m_num_nodes - 1);

        for (const std::string& participant_name : dds_communicator->GetMatchedParticipantNames())
            RegisterParticipant(m_communicator, participant_name);
    }
#endif

    // Gather all of the underlying messages and add those to the communicator
    auto descriptions = GatherDescriptionMessages();
    m_communicator->AddOutgoingMessages(descriptions);

    // Send the description messages out to each node and receive any other messages
    m_communicator->Synchronize();

    // Process any received data
    // If first pass, data will typically contain description messages that describe new agents
    // Otherwise, will contain state or general purpose messages
    ProcessReceivedMessages();

    // Create agents from received descriptions
    CreateAgentsFromDescriptions();

    // Initialize each zombie with the passed system
    for (auto& zombie_pair : m_zombies)
        zombie_pair.second->InitializeZombie(system);

    m_initialized = true;

    return true;
}

void SynChronoManager::Synchronize(double time) {
    if (!m_communicator)
        return;

    // If time to next sync is in the future, do nothing
    if (time < m_next_sync)
        return;

    // Call update for each underlying agent
    UpdateAgents();

    // Gather messages from each node and add those to the communicator
    // Only add the messages to the communicator which is responsible for commuticating with that node
    SynMessageList messages = GatherMessages();
    m_communicator->AddOutgoingMessages(messages);

    // Send the messages out to each node and receive any other messages
    m_communicator->Synchronize();

    // Process any received data
    // Will most likely contain state or general purpose messages
    ProcessReceivedMessages();

    // Distribute the organized messages
    DistributeMessages();

    // Reset
    m_communicator->Reset();     // Reset the communicator
    m_messages.clear();          // clean the message map
    m_next_sync += m_heartbeat;  // Set next sync to a point in the future
}

void SynChronoManager::UpdateAgents() {
    for (auto& agent_pair : m_agents)
        agent_pair.second->Update();
}

// --------------------------------------------------------------------------------------------------------------

SynMessageList SynChronoManager::GatherMessages() {
    SynMessageList messages;

    // Gather messages from each agent on this node
    for (auto& agent_pair : m_agents)
        agent_pair.second->GatherMessages(messages);

    return messages;
}

SynMessageList SynChronoManager::GatherDescriptionMessages() {
    SynMessageList messages;

    // Gather the description messages from each agent on this node
    for (auto& agent_pair : m_agents)
        agent_pair.second->GatherDescriptionMessages(messages);

    return messages;
}

void SynChronoManager::ProcessReceivedMessages() {
    // get the message buffer from the underlying communicator
    SynMessageList messages = m_communicator->GetMessages();

    for (auto message : messages) {
        /// Check the messages intended destination
        SynAgentID destination_id = message->GetDestinationID();

        /// TODO: Sort into the intended destination
        for (auto& agent_pair : m_agents) {
            m_messages[agent_pair.second].push_back(message);
        }
    }
}

void SynChronoManager::DistributeMessages() {
    for (auto& message_agent_pair : m_messages) {
        // For readibility
        auto to_agent = message_agent_pair.first;
        auto messages = message_agent_pair.second;

        for (auto message : messages) {
            auto from_zombie = m_zombies[message->GetSourceID()];

            // std::map will return a nullptr if zombie does not exist (i.e. has never been added)
            if (!from_zombie) {
                SynLog() << "The intended agent (" << message->GetSourceID() << ") is not on this node!\n";
                continue;
            }

            // SynLog() << "Processing message from zombie with ID " << message->GetSourceID() << "\n";

            from_zombie->SynchronizeZombie(message);
            to_agent->ProcessMessage(message);
        }
    }
}

void SynChronoManager::CreateAgentsFromDescriptions() {
    for (auto& message_agent_pair : m_messages) {
        // For readibility
        auto to_agent = message_agent_pair.first;
        auto messages = message_agent_pair.second;

        // Will remove the message if used
        for (SynMessageList::iterator it = messages.begin(); it != messages.end();) {
            // Try to create a zombie from the passed message
            auto message = *it;
            try {
                /// TODO: In the future, this shouldn't be necessary since destination and source will be used
                if (auto zombie = m_zombies[message->GetSourceID()]) {
                    to_agent->RegisterZombie(zombie);
                } else {
                    zombie = SynAgentFactory::CreateAgent(message);
                    AddZombie(zombie, message->GetSourceID());
                    to_agent->RegisterZombie(zombie);

                    // SynLog() << "Added agent with ID " << message->GetSourceID() << "\n";
                }
                messages.erase(it);

            } catch (ChException err) {
                ++it;
            }
        }
    }
}

}  // namespace synchrono
}  // namespace chrono