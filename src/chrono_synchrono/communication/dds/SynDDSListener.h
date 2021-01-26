#ifndef SYN_DDS_LISTENER_H
#define SYN_DDS_LISTENER_H

#include "chrono_synchrono/SynApi.h"

#undef ALIVE

#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>

#include <mutex>
#include <string>
#include <functional>
#include <condition_variable>

namespace chrono {
namespace synchrono {

/// @addtogroup synchrono_communication_dds
/// @{

/// Thread safe counter used to count various elements
/// Use case would be to block until a certain number of elements. Doesn't busy wait but uses mutex instead.
class SynDDSMatchCounter {
  public:
    SynDDSMatchCounter() : m_iters(0) {}

    void BlockUntil(int iters);

    void Increment();

    void SetSafe(int iters);

  private:
    int m_iters;
    std::mutex m_mutex;
    std::condition_variable m_condition_variable;
};

// -----------------------------------------------------------------------------------

/// Participant listener that will count the number of participants and store their names to be used later
class SynDDSParticipantListener : public eprosima::fastdds::dds::DomainParticipantListener {
  public:
    SynDDSParticipantListener() {}

    virtual void on_participant_discovery(eprosima::fastdds::dds::DomainParticipant* participant,
                                          eprosima::fastrtps::rtps::ParticipantDiscoveryInfo&& info) override;

    ///@brief Wait for the specified number of matches
    /// Each subscriber listener has a callback that will be called when a subscriber is matched with
    /// a DataWriter. This function blocks until that the matches are acheived. By default,
    /// a subscriber will just wait for a single listener.
    ///
    void BlockUntilMatches(unsigned int matches) { m_match_counter.BlockUntil(matches); }

    ///@brief Get the names of matched participants
    /// When a participant is matched, in the SynChrono world, the participant name
    /// is used to designate from which node the participant is responsible for. A
    /// subscriber is then created to listen to those messages
    ///
    ///@return std::vector<std::string>
    std::vector<std::string> GetParticipantNames() const { return m_participant_names; }

  private:
    SynDDSMatchCounter m_match_counter;

    std::vector<std::string> m_participant_names;
};

// -----------------------------------------------------------------------------------

/// Data writer listener that counts number of subscribers listening to a specific topic
class SynDDSDataWriterListener : public eprosima::fastdds::dds::DataWriterListener {
  public:
    SynDDSDataWriterListener() {}

    virtual void on_publication_matched(eprosima::fastdds::dds::DataWriter* writer,
                                        const eprosima::fastdds::dds::PublicationMatchedStatus& info) override;

    ///@brief Wait for the specified number of matches
    /// Each subscriber listener has a callback that will be called when a subscriber is matched with
    /// a DataWriter. This function blocks until that the matches are acheived. By default,
    /// a subscriber will just wait for a single listener.
    ///
    void BlockUntilMatches(unsigned int matches) { m_match_counter.BlockUntil(matches); }

  private:
    SynDDSMatchCounter m_match_counter;
};

// -----------------------------------------------------------------------------------

/// Data reader listener that can both count publishers on a topic and can be used as an asynchronous listener
class SynDDSDataReaderListener : public eprosima::fastdds::dds::DataReaderListener {
  public:
    SynDDSDataReaderListener() {}
    SynDDSDataReaderListener(std::function<void(void*)> callback, void* message);

    virtual void on_data_available(eprosima::fastdds::dds::DataReader* reader) override;
    virtual void on_subscription_matched(eprosima::fastdds::dds::DataReader* reader,
                                         const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) override;

    void SetOnDataAvailableCallback(std::function<void(void*)> callback);

    ///@brief Wait for the specified number of matches
    /// Each subscriber listener has a callback that will be called when a subscriber is matched with
    /// a DataWriter. This function blocks until that the matches are acheived. By default,
    /// a subscriber will just wait for a single listener.
    ///
    void BlockUntilMatches(unsigned int matches) { m_match_counter.BlockUntil(matches); }

  private:
    SynDDSMatchCounter m_match_counter;

    void* m_message;
    std::function<void(void*)> m_on_data_available_callback;
};

/// @} synchrono_communication

}  // namespace synchrono
}  // namespace chrono

#endif