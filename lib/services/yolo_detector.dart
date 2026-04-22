import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:graduation_project/services/ttf_service.dart';
import 'package:image/image.dart' as imgLib;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'dart:math' as math;

class YoloDetector {
  late tfl.Interpreter _interpreter;
  final TtsService tts;

  YoloDetector(this.tts);

  List<String> _labels = [];
  bool _isProcessing = false;

  // ================== SPEECH CONTROL ==================
  DateTime _lastSpokenTime = DateTime.fromMillisecondsSinceEpoch(0);
  String _lastSpokenText = "";
  static const Duration SPEECH_DELAY = Duration(seconds: 3);
  // ====================================================

  // ================== CONFIG ==================
  static const int INPUT_SIZE   = 640;
  static const double CONF_THRESHOLD = 0.25;
  static const int MAX_DETECTIONS = 300; // output rows
  // ============================================
  // Output shape: [1, 300, 6]
  // Each row: [x1, y1, x2, y2, confidence, class_id]
  // Coordinates are normalized 0-1 relative to 640x640 canvas
  // NMS already applied by the model

  // ================== LOAD MODEL ==================
  Future<void> loadModel() async {
    _interpreter = await tfl.Interpreter.fromAsset(
      'assets/models/best_float32.tflite',
    );
    _interpreter.allocateTensors();

    print("Input  shape: ${_interpreter.getInputTensor(0).shape}");
    print("Output shape: ${_interpreter.getOutputTensor(0).shape}");

    await loadLabels();

    await _safeSpeak(
      tts.isArabic
          ? "مرحبا، أنا مساعدك البصري، جاهز"
          : "Hello, I am your visual assistant, ready",
    );
  }

  Future<void> loadLabels() async {
    final path = tts.isArabic
        ? 'assets/labels/labels_arabic_new.txt'
        : 'assets/labels/labels_new.txt';

    final data = await rootBundle.loadString(path);

    // Format: "0 person", "1 bicycle", ...
    _labels = data
        .split('\n')
        .where((e) => e.trim().isNotEmpty)
        .map((e) => e.trim().split(' ').skip(1).join(' '))
        .toList();

    print("Loaded ${_labels.length} labels.");
  }

  // ================== DETECT FROM CAMERA FRAME ==================
  Future<void> detectFrame(CameraImage image) async {
    if (_isProcessing) return;
    _isProcessing = true;

    final rgb = _convertYUV420(image);
    if (rgb == null) { _isProcessing = false; return; }

    final result = _runInference(_preprocessImage(rgb));

    if (result == null || result.isEmpty) {
      await _safeSpeak(tts.isArabic ? "لا يوجد شيء واضح" : "Nothing detected");
    } else {
      await _safeSpeak(
        tts.isArabic ? _buildArabicSpeech(result) : _buildEnglishSpeech(result),
      );
    }

    _isProcessing = false;
  }

  // ================== DETECT FROM STILL IMAGE ==================
  Future<String?> detectImage(Uint8List bytes) async {
    if (_isProcessing) return null;
    _isProcessing = true;

    try {
      final img = imgLib.decodeImage(bytes);
      if (img == null) {
        _isProcessing = false;
        return tts.isArabic ? "فشل في قراءة الصورة" : "Failed to read image";
      }

      final input  = _preprocessImage(img);
      final result = _runInference(input);

      _isProcessing = false;

      if (result == null || result.isEmpty) {
        return tts.isArabic ? "لا يوجد شيء واضح" : "Nothing detected";
      }

      return tts.isArabic
          ? _buildArabicSpeech(result)
          : _buildEnglishSpeech(result);
    } catch (e) {
      print("detectImage error: $e");
      _isProcessing = false;
      return tts.isArabic ? "حدث خطأ" : "Error occurred";
    }
  }

  // ================== INFERENCE ==================
  /// Returns Map<labelName, count> or null if nothing found
  Map<String, int>? _runInference(Float32List input) {
    // Output: [1, 300, 6]
    final output = List.generate(
      1,
          (_) => List.generate(MAX_DETECTIONS, (_) => List.filled(6, 0.0)),
    );

    _interpreter.run(input.buffer.asUint8List(), output);

    final Map<String, int> count = {};

    for (int i = 0; i < MAX_DETECTIONS; i++) {
      final row        = output[0][i];
      final confidence = row[4];
      final classId    = row[5].round();

      if (confidence < CONF_THRESHOLD) continue;
      if (classId < 0 || classId >= _labels.length) continue;

      final label = _labels[classId];
      count[label] = (count[label] ?? 0) + 1;
    }

    return count.isEmpty ? null : count;
  }

  // ================== PREPROCESS ==================
  Float32List _preprocessImage(imgLib.Image image) {
    final h     = image.height;
    final w     = image.width;
    final scale = INPUT_SIZE / math.max(h, w);

    final resized = imgLib.copyResize(
      image,
      width:  (w * scale).round(),
      height: (h * scale).round(),
    );

    final canvas = imgLib.Image(width: INPUT_SIZE, height: INPUT_SIZE);
    imgLib.fill(canvas, color: imgLib.ColorRgb8(114, 114, 114));

    final dx = (INPUT_SIZE - resized.width)  ~/ 2;
    final dy = (INPUT_SIZE - resized.height) ~/ 2;
    imgLib.compositeImage(canvas, resized, dstX: dx, dstY: dy);

    final input = Float32List(INPUT_SIZE * INPUT_SIZE * 3);
    int index = 0;

    for (int y = 0; y < INPUT_SIZE; y++) {
      for (int x = 0; x < INPUT_SIZE; x++) {
        final p = canvas.getPixel(x, y);
        input[index++] = p.r / 255.0;
        input[index++] = p.g / 255.0;
        input[index++] = p.b / 255.0;
      }
    }

    return input;
  }

  // ================== YUV → RGB ==================
  imgLib.Image? _convertYUV420(CameraImage image) {
    final width  = image.width;
    final height = image.height;

    final yPlane = image.planes[0].bytes;
    final uPlane = image.planes[1].bytes;
    final vPlane = image.planes[2].bytes;

    final yStride      = image.planes[0].bytesPerRow;
    final uvStride     = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 2;

    final img = imgLib.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      final yOffset  = y * yStride;
      final uvOffset = (y ~/ 2) * uvStride;

      for (int x = 0; x < width; x++) {
        final uvIndex = uvOffset + (x ~/ 2) * uvPixelStride;

        final yp = yPlane[yOffset + x] & 0xff;
        final up = (uPlane[uvIndex] & 0xff) - 128;
        final vp = (vPlane[uvIndex] & 0xff) - 128;

        final r = (yp + 1.402 * vp).round().clamp(0, 255);
        final g = (yp - 0.344 * up - 0.714 * vp).round().clamp(0, 255);
        final b = (yp + 1.772 * up).round().clamp(0, 255);

        img.setPixelRgb(x, y, r, g, b);
      }
    }
    return img;
  }

  // ================== SAFE SPEAK ==================
  Future<void> _safeSpeak(String text) async {
    final now = DateTime.now();

    if (text == _lastSpokenText &&
        now.difference(_lastSpokenTime) < SPEECH_DELAY) return;

    if (now.difference(_lastSpokenTime) < SPEECH_DELAY) return;

    _lastSpokenText = text;
    _lastSpokenTime = now;

    await tts.speak(text);
  }

  // ================== SPEECH BUILD ==================
  String _buildArabicSpeech(Map<String, int> count) {
    final parts = <String>[];
    count.forEach((k, v) => parts.add(v == 1 ? "واحد $k" : "$v $k"));
    return "يوجد ${parts.join(' و ')}";
  }

  String _buildEnglishSpeech(Map<String, int> count) {
    final parts = <String>[];
    count.forEach((k, v) => parts.add("$v $k${v > 1 ? 's' : ''}"));
    return "I see ${parts.join(' and ')}";
  }

  // ================== DISPOSE ==================
  void dispose() {
    _interpreter.close();
  }
}