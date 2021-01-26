#include "chrono_synchrono/communication/dds/SynDDSListener.h"

#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastrtps::rtps;

namespace chrono {
namespace synchrono {

void SynDDSMatchCounter::BlockUntil(int iters) {
    std::unique_lock<std::mutex> lock(m_mutex);
    while (iters > m_iters)
        m_condition_variable.wait(lock);
}

void SynDDSMatchCounter::Increment() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_iters++;

    m_condition_variable.notify_all();
}

void SynDDSMatchCounter::SetSafe(int iters) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_iters = iters;

    m_condition_variable.notify_all();
}

// -----------------------------------------------------------------------------------

void SynDDSParticipantListener::on_participant_discovery(DomainParticipant* participant,
                                                         ParticipantDiscoveryInfo&& info) {
    if (info.status == ParticipantDiscoveryInfo::DISCOVERED_PARTICIPANT) {
        m_participant_names.push_back(std::string(info.info.m_participantName));
        m_match_counter.Increment();
    } else if (info.status == ParticipantDiscoveryInfo::REMOVED_PARTICIPANT ||
               info.status == ParticipantDiscoveryInfo::DROPPED_PARTICIPANT) {
    }
}

// -----------------------------------------------------------------------------------

void SynDDSDataWriterListener::on_publication_matched(DataWriter* writer, const PublicationMatchedStatus& info) {
    if (info.current_count_change == 1 || info.current_count_change == -1) {
        m_match_counter.SetSafe(info.total_count);
    } else {
        std::cout << info.current_count_change
                  << " is not a valid value for PublicationMatchedStatus "
                     "current count change."
                  << std::endl;
    }
}

// -----------------------------------------------------------------------------------

SynDDSDataReaderListener::SynDDSDataReaderListener(std::function<void(void*)> callback, void* message)
    : m_on_data_available_callback(callback), m_message(message) {}

void SynDDSDataReaderListener::on_data_available(DataReader* reader) {
    // std::cout << "Data Available" << std::endl;

    SampleInfo info;
    if (reader->take_next_sample(m_message, &info) != ReturnCode_t::RETCODE_OK) {
        std::cout << "SynDDSDataReaderListener::on_data_available: DataReader failed to read message." << std::endl;
        exit(-1);
    }

    m_on_data_available_callback(m_message);
}

void SynDDSDataReaderListener::on_subscription_matched(DataReader* reader, const SubscriptionMatchedStatus& info) {
    if (info.current_count_change == 1 || info.current_count_change == -1) {
        m_match_counter.SetSafe(info.total_count);
    } else {
        std::cout << info.current_count_change
                  << " is not a valid value for SubscriptionMatchedStatus "
                     "current count change"
                  << std::endl;
    }
}

void SynDDSDataReaderListener::SetOnDataAvailableCallback(std::function<void(void*)> callback) {
    m_on_data_available_callback = callback;
}

}  // namespace synchrono
}  // namespace chrono