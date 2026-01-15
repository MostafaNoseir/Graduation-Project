import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:graduation_project/services/ttf_service.dart';
import 'package:image/image.dart' as imgLib;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

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
  static const int INPUT_SIZE = 640;
  static const double CONF_THRESHOLD = 0.25;
  static const double IOU_THRESHOLD = 0.45;
  static const int NUM_CLASSES = 80;
  static const int NUM_ANCHORS = 8400;
  // ============================================

  // ================== LOAD MODEL ==================
  Future<void> loadModel() async {
    _interpreter =
    await tfl.Interpreter.fromAsset('assets/models/yolo11n_float32.tflite');

    await loadLabels();

    await _safeSpeak(
      tts.isArabic
          ? "مرحبا، أنا مساعدك البصري، جاهز"
          : "Hello, I am your visual assistant, ready",
    );
  }

  Future<void> loadLabels() async {
    final path = tts.isArabic
        ? 'assets/labels/labels_arabic.txt'
        : 'assets/labels/labels.txt';

    final data = await rootBundle.loadString(path);
    _labels = data
        .split('\n')
        .where((e) => e.trim().isNotEmpty)
        .map((e) => e.split(' ').skip(1).join(' '))
        .toList();
  }

  // ================== MAIN DETECTION ==================
  Future<void> detectFrame(CameraImage image) async {
    if (_isProcessing) return;
    _isProcessing = true;

    final input = _preprocess(image);
    if (input.isEmpty) {
      _isProcessing = false;
      return;
    }

    final output =
    List.filled(1 * 84 * NUM_ANCHORS, 0.0).reshape([1, 84, NUM_ANCHORS]);

    _interpreter.run(input.buffer.asUint8List(), output);

    final boxes = <List<double>>[];
    final scores = <double>[];
    final classIds = <int>[];

    for (int i = 0; i < NUM_ANCHORS; i++) {
      final cx = output[0][0][i];
      final cy = output[0][1][i];
      final w = output[0][2][i];
      final h = output[0][3][i];

      double maxScore = 0;
      int classId = 0;

      for (int c = 0; c < NUM_CLASSES; c++) {
        final score = output[0][4 + c][i];
        if (score > maxScore) {
          maxScore = score;
          classId = c;
        }
      }

      if (maxScore < CONF_THRESHOLD) continue;

      boxes.add([
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2,
      ]);

      scores.add(maxScore);
      classIds.add(classId);
    }

    if (boxes.isEmpty) {
      await _safeSpeak(
        tts.isArabic ? "لا يوجد شيء واضح" : "Nothing detected",
      );
      _isProcessing = false;
      return;
    }

    final keep = _nms(boxes, scores);

    final Map<String, int> count = {};
    for (final i in keep) {
      final label = _labels[classIds[i]];
      count[label] = (count[label] ?? 0) + 1;
    }

    final speech = tts.isArabic
        ? _buildArabicSpeech(count)
        : _buildEnglishSpeech(count);

    await _safeSpeak(speech);

    _isProcessing = false;
  }
  Future<String?> detectImage(Uint8List bytes) async {
    if (_isProcessing) return null;
    _isProcessing = true;

    try {
      final img = imgLib.decodeImage(bytes);
      if (img == null) {
        _isProcessing = false;
        return tts.isArabic ? "فشل في قراءة الصورة" : "Failed to read image";
      }

      final input = _preprocessImage(img);
      if (input.isEmpty) {
        _isProcessing = false;
        return null;
      }

      final output =
      List.filled(1 * 84 * NUM_ANCHORS, 0.0).reshape([1, 84, NUM_ANCHORS]);
      _interpreter.run(input.buffer.asUint8List(), output);

      final boxes = <List<double>>[];
      final scores = <double>[];
      final classIds = <int>[];

      for (int i = 0; i < NUM_ANCHORS; i++) {
        final cx = output[0][0][i];
        final cy = output[0][1][i];
        final w = output[0][2][i];
        final h = output[0][3][i];

        double maxScore = 0;
        int classId = 0;

        for (int c = 0; c < NUM_CLASSES; c++) {
          final score = output[0][4 + c][i];
          if (score > maxScore) {
            maxScore = score;
            classId = c;
          }
        }

        if (maxScore < CONF_THRESHOLD) continue;

        boxes.add([
          cx - w / 2,
          cy - h / 2,
          cx + w / 2,
          cy + h / 2,
        ]);

        scores.add(maxScore);
        classIds.add(classId);
      }

      if (boxes.isEmpty) {
        _isProcessing = false;
        return tts.isArabic ? "لا يوجد شيء واضح" : "Nothing detected";
      }

      final keep = _nms(boxes, scores);

      final Map<String, int> count = {};
      for (final i in keep) {
        final label = _labels[classIds[i]];
        count[label] = (count[label] ?? 0) + 1;
      }

      _isProcessing = false;

      return tts.isArabic
          ? _buildArabicSpeech(count)
          : _buildEnglishSpeech(count);
    } catch (e) {
      _isProcessing = false;
      return tts.isArabic ? "حدث خطأ" : "Error occurred";
    }
  }

  // ================== SAFE SPEAK (3s DELAY) ==================
  Future<void> _safeSpeak(String text) async {
    final now = DateTime.now();

    if (text == _lastSpokenText &&
        now.difference(_lastSpokenTime) < SPEECH_DELAY) {
      return;
    }

    if (now.difference(_lastSpokenTime) < SPEECH_DELAY) {
      return;
    }

    _lastSpokenText = text;
    _lastSpokenTime = now;

    await tts.speak(text);
  }

  // ================== IMAGE PREPROCESS ==================
  Float32List _preprocess(CameraImage image) {
    final rgb = _convertYUV420(image);
    if (rgb == null) return Float32List(0);

    final h = rgb.height;
    final w = rgb.width;
    final scale = INPUT_SIZE / math.max(h, w);

    final resized = imgLib.copyResize(
      rgb,
      width: (w * scale).round(),
      height: (h * scale).round(),
    );

    final canvas = imgLib.Image(width: INPUT_SIZE, height: INPUT_SIZE);
    imgLib.fill(canvas, color: imgLib.ColorRgb8(114, 114, 114));

    final dx = (INPUT_SIZE - resized.width) ~/ 2;
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
  Float32List _preprocessImage(imgLib.Image image) {
    final h = image.height;
    final w = image.width;
    final scale = INPUT_SIZE / math.max(h, w);

    final resized = imgLib.copyResize(
      image,
      width: (w * scale).round(),
      height: (h * scale).round(),
    );

    final canvas = imgLib.Image(width: INPUT_SIZE, height: INPUT_SIZE);
    imgLib.fill(canvas, color: imgLib.ColorRgb8(114, 114, 114));

    final dx = (INPUT_SIZE - resized.width) ~/ 2;
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


  imgLib.Image? _convertYUV420(CameraImage image) {
    final width = image.width;
    final height = image.height;

    final yPlane = image.planes[0].bytes;
    final uPlane = image.planes[1].bytes;
    final vPlane = image.planes[2].bytes;

    final yStride = image.planes[0].bytesPerRow;
    final uvStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 2;

    final img = imgLib.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      final yOffset = y * yStride;
      final uvOffset = (y ~/ 2) * uvStride;

      for (int x = 0; x < width; x++) {
        final uvIndex = uvOffset + (x ~/ 2) * uvPixelStride;

        final yp = yPlane[yOffset + x] & 0xff;
        final up = (uPlane[uvIndex] & 0xff) - 128;
        final vp = (vPlane[uvIndex] & 0xff) - 128;

        final r = (yp + 1.402 * vp).round().clamp(0, 255);
        final g =
        (yp - 0.344 * up - 0.714 * vp).round().clamp(0, 255);
        final b = (yp + 1.772 * up).round().clamp(0, 255);

        img.setPixelRgb(x, y, r, g, b);
      }
    }
    return img;
  }

  // ================== NMS ==================
  List<int> _nms(List<List<double>> boxes, List<double> scores) {
    final order = List.generate(scores.length, (i) => i)
      ..sort((a, b) => scores[b].compareTo(scores[a]));

    final suppressed = List<bool>.filled(scores.length, false);
    final keep = <int>[];

    for (final i in order) {
      if (suppressed[i]) continue;
      keep.add(i);

      for (int j = 0; j < boxes.length; j++) {
        if (i == j || suppressed[j]) continue;
        if (_iou(boxes[i], boxes[j]) > IOU_THRESHOLD) {
          suppressed[j] = true;
        }
      }
    }
    return keep;
  }

  double _iou(List<double> a, List<double> b) {
    final x1 = math.max(a[0], b[0]);
    final y1 = math.max(a[1], b[1]);
    final x2 = math.min(a[2], b[2]);
    final y2 = math.min(a[3], b[3]);

    if (x2 <= x1 || y2 <= y1) return 0;

    final inter = (x2 - x1) * (y2 - y1);
    final areaA = (a[2] - a[0]) * (a[3] - a[1]);
    final areaB = (b[2] - b[0]) * (b[3] - b[1]);

    return inter / (areaA + areaB - inter);
  }

  // ================== SPEECH BUILD ==================
  String _buildArabicSpeech(Map<String, int> count) {
    final parts = <String>[];
    count.forEach((k, v) {
      parts.add(v == 1 ? "واحد $k" : "$v $k");
    });
    return "يوجد ${parts.join(' و ')}";
  }

  String _buildEnglishSpeech(Map<String, int> count) {
    final parts = <String>[];
    count.forEach((k, v) {
      parts.add("$v $k${v > 1 ? 's' : ''}");
    });
    return "I see ${parts.join(' and ')}";
  }

  // ================== DISPOSE ==================
  void dispose() {
    _interpreter.close();
  }
}
