# Changelog

All notable changes to ComfyUI-NoiseGen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-26

### Added
- Initial release of ComfyUI-NoiseGen
- Universal NoiseGenerator node with 7 noise types
- Dedicated nodes for each noise type (White, Pink, Brown, Blue, Violet, Perlin, Band-Limited)
- Audio save functionality with SaveAudioAdvanced node
- Comprehensive stereo and multi-channel support (1-8 channels)
- Three stereo modes: independent, correlated, decorrelated
- Stereo width control (0.0-2.0 range)
- Professional-grade audio quality with 32-bit float precision
- Scientific-grade noise generation algorithms
- Duration control (0.1-300 seconds)
- Multiple sample rate support (8kHz-96kHz)
- Amplitude control with proper normalization
- Seed-based reproducibility
- Example workflows for common use cases
- Comprehensive test suite

### Features
- **Noise Types**: White, Pink, Brown, Blue, Violet, Perlin, Band-Limited
- **Audio Quality**: 32-bit float, professional sample rates
- **Stereo Support**: Full multi-channel with width control
- **Parameter Control**: Duration, amplitude, seed, sample rate
- **ComfyUI Integration**: Native audio format support
- **Example Workflows**: Ready-to-use templates

### Documentation
- Complete README with installation and usage instructions
- Example workflow files with detailed descriptions
- Comprehensive API documentation
- Technical specifications for each noise type 