#pragma once

#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>

namespace m_engine::executor {

class TrtLogger : public nvinfer1::ILogger {
 public:
  static TrtLogger* GetTrtLogger();

  void log(nvinfer1::ILogger::Severity severity,
           const char* msg) noexcept override;

 private:
  TrtLogger() = default;
};  // class Logger

}  // namespace m_engine::executor
