#include "idl/imu_framePubSubTypes.hpp"
#include <cstdio>
#include <cstddef>
int main(){
    using rc::media::ImuFrame;
    ImuFrame f{};
    rc::media::ImuFramePubSubType t;
    const char* base = reinterpret_cast<const char*>(&f);
    const size_t off_gyro = reinterpret_cast<const char*>(&f.gyro_var()) - base;
    const size_t off_acc  = reinterpret_cast<const char*>(&f.acc_var())  - base;
    std::printf("sizeof(ImuFrame)      = %zu\n", sizeof(ImuFrame));
    std::printf("offsetof(gyro_var)    = %zu\n", off_gyro);
    std::printf("offsetof(acc_var)     = %zu   (+4 = %zu, the is_plain constant)\n", off_acc, off_acc+4);
    const bool plain = t.is_plain(eprosima::fastdds::dds::DataRepresentationId_t::XCDR2_DATA_REPRESENTATION);
    std::printf("is_plain(XCDR2)       = %s\n", plain ? "TRUE  -> zero-copy loan kept" : "FALSE -> LOAN LOST, every sample serialized");
    return plain ? 0 : 1;
}
