#include "m_engine/executor/trt_logger.h"

#include <NvInferRuntimeCommon.h>
#include <glog/logging.h>

namespace m_engine::executor {

void TrtLogger::log(Severity severity, const char* msg) noexcept {
  switch (severity) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
      // Internal error has occurred, execution is unrecoverable.
      LOG(FATAL) << "[TRT internal error]: " << msg;
      break;
    case nvinfer1::ILogger::Severity::kERROR:
      // Application error has occurred.
      LOG(ERROR) << "[TRT E]: " << msg;
      break;
    case nvinfer1::ILogger::Severity::kWARNING:
      // Application error has been discovered. TRT has recovered or fallen
      // back to a default.
      LOG(WARNING) << "[TRT W]: " << msg;
      break;
    case nvinfer1::ILogger::Severity::kINFO:
      // Informational messages with instructional information.
      LOG(INFO) << "[TRT I]: " << msg;
      break;
    case nvinfer1::ILogger::Severity::kVERBOSE:
      // Verbose message with debugging information.
      VLOG(1) << "[TRT V]: " << msg;
      break;
    default:
      LOG(FATAL) << "Unknown TensorRT logging severity: " << msg;
      break;
  }
}

TrtLogger* TrtLogger::GetTrtLogger() {
  static TrtLogger logger;
  return &logger;
}

}  // namespace m_engine::executor
