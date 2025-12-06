// lib/yolo_detector.dart
import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:io' show Platform;

import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:camera/camera.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as imgLib;

class YoloDetector extends ChangeNotifier {
  late tfl.Interpreter _interpreter;
  List<String> _labels = [];
  late FlutterTts _tts;

  static const int INPUT_SIZE = 640;
  static const double CONF_THRESHOLD = 0.25;
  static const double IOU_THRESHOLD = 0.45;

  bool _isSpeaking = false;
  bool _isArabic = true;

  bool get isArabic => _isArabic;

  Future<void> loadModel() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset('assets/models/yolo11n_float32.tflite');

      await _loadLabels(_isArabic);

      _tts = FlutterTts();
      await _tts.awaitSpeakCompletion(true);
      await _tts.setVolume(1.0);
      await _tts.setSpeechRate(0.5);
      await _tts.setPitch(1.0);

      if (Platform.isAndroid) {
        await _tts.setEngine("com.google.android.tts");
      }

      await _updateTtsLanguage();

      print("تم تحميل النموذج بنجاح");
      await _tts.speak(_isArabic
          ? "مرحبا، أنا مساعدك البصري، جاهز لوصف ما أراه"
          : "Hello, I'm your visual assistant, ready to describe what I see");

      notifyListeners();
    } catch (e) {
      print("خطأ في تحميل النموذج: $e");
    }
  }

  Future<void> _loadLabels(bool arabic) async {
    final path = arabic
        ? 'assets/labels/labels_arabic.txt'
        : 'assets/labels/labels.txt';
    final data = await rootBundle.loadString(path);
    _labels = data
        .split('\n')
        .where((line) => line.trim().isNotEmpty)
        .map((line) => line.split(' ').skip(1).join(' '))
        .toList();
  }

  Future<void> toggleLanguage() async {
    _isArabic = !_isArabic;
    await _loadLabels(_isArabic);
    await _updateTtsLanguage();
    await _tts.speak(_isArabic ? "تم التبديل إلى العربية" : "Switched to English");
    notifyListeners();
  }

  Future<void> _updateTtsLanguage() async {
    await _tts.setLanguage(_isArabic ? "ar" : "en-US");
  }

  // تحويل YUV420 إلى RGB
  imgLib.Image? _convertYUV420ToImage(CameraImage cameraImage) {
    final width = cameraImage.width;
    final height = cameraImage.height;

    final yPlane = cameraImage.planes[0].bytes;
    final uPlane = cameraImage.planes[1].bytes;
    final vPlane = cameraImage.planes[2].bytes;

    final yRowStride = cameraImage.planes[0].bytesPerRow;
    final uvRowStride = cameraImage.planes[1].bytesPerRow;
    final uvPixelStride = cameraImage.planes[1].bytesPerPixel ?? 2;

    final image = imgLib.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      final yOffset = y * yRowStride;
      final uvOffset = (y ~/ 2) * uvRowStride;
      for (int x = 0; x < width; x++) {
        final uvIndex = uvOffset + (x ~/ 2) * uvPixelStride;

        final yp = yPlane[yOffset + x] & 0xff;
        final up = (uPlane[uvIndex] & 0xff) - 128;
        final vp = (vPlane[uvIndex] & 0xff) - 128;

        int r = (yp + 1.402 * vp).round().clamp(0, 255);
        int g = (yp - 0.344136 * up - 0.714136 * vp).round().clamp(0, 255);
        int b = (yp + 1.772 * up).round().clamp(0, 255);

        image.setPixelRgb(x, y, r, g, b);
      }
    }
    return image;
  }

  // معالجة الصورة (Letterbox + Normalize)
  Float32List _preprocess(CameraImage image) {
    final img = _convertYUV420ToImage(image);
    if (img == null) return Float32List(0);

    final h = img.height;
    final w = img.width;
    final scale = INPUT_SIZE / math.max(h, w);
    final newH = (h * scale).round();
    final newW = (w * scale).round();

    final resized = imgLib.copyResize(img, width: newW, height: newH);

    final letterbox = imgLib.Image(width: INPUT_SIZE, height: INPUT_SIZE);
    imgLib.fill(letterbox, color: imgLib.ColorRgb8(114, 114, 114));
    final offsetX = (INPUT_SIZE - newW) ~/ 2;
    final offsetY = (INPUT_SIZE - newH) ~/ 2;
    imgLib.compositeImage(letterbox, resized, dstX: offsetX, dstY: offsetY);

    final input = Float32List(1 * INPUT_SIZE * INPUT_SIZE * 3);
    int idx = 0;
    for (int y = 0; y < INPUT_SIZE; y++) {
      for (int x = 0; x < INPUT_SIZE; x++) {
        final pixel = letterbox.getPixel(x, y);
        input[idx++] = pixel.r / 255.0;
        input[idx++] = pixel.g / 255.0;
        input[idx++] = pixel.b / 255.0;
      }
    }
    return input;
  }

  // الكشف الرئيسي
  Future<void> detectFrame(CameraImage image) async {
    if (_isSpeaking) return;

    final input = _preprocess(image);
    if (input.isEmpty) return;

    final output = List.filled(1 * 84 * 8400, 0.0).reshape([1, 84, 8400]);
    _interpreter.run(input.buffer.asUint8List(), output);

    final boxes = <List<double>>[];
    final scores = <double>[];
    final classIds = <int>[];

    for (int i = 0; i < 8400; i++) {
      final cx = output[0][0][i];
      final cy = output[0][1][i];
      final w  = output[0][2][i];
      final h  = output[0][3][i];

      final classScores = List.generate(80, (c) => output[0][4 + c][i]);

      final maxScore = classScores.reduce((a, b) => a > b ? a : b);
      final maxClassId = classScores.indexOf(maxScore);

      if (maxScore < CONF_THRESHOLD) continue;

      boxes.add([cx - w/2, cy - h/2, cx + w/2, cy + h/2]);
      scores.add(maxScore);
      classIds.add(maxClassId);
    }

    if (boxes.isEmpty) {
      await _speak(_isArabic ? "لا يوجد شيء واضح" : "Nothing detected");
      return;
    }

    final indices = _nms(boxes, scores);

    final count = <String, int>{};
    for (final i in indices) {
      final label = _labels[classIds[i]];
      count[label] = (count[label] ?? 0) + 1;
    }

    final speech = _isArabic
        ? _buildArabicSpeech(count)
        : _buildEnglishSpeech(count);

    await _speak(speech);
  }

  // NMS

  List<int> _nms(List<List<double>> boxes, List<double> scores) {
    if (boxes.isEmpty) return [];

    final indices = List.generate(scores.length, (i) => i);
    indices.sort((a, b) => scores[b].compareTo(scores[a]));

    final suppressed = List.filled(boxes.length, false);
    final keep = <int>[];

    for (final i in indices) {
      if (suppressed[i]) continue;
      keep.add(i);
      for (int j = 0; j < boxes.length; j++) {
        if (suppressed[j] || i == j) continue;
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
    if (x2 <= x1 || y2 <= y1) return 0.0;
    final inter = (x2 - x1) * (y2 - y1);
    final areaA = (a[2] - a[0]) * (a[3] - a[1]);
    final areaB = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (areaA + areaB - inter);
  }

  String _buildArabicSpeech(Map<String, int> count) {
    if (count.isEmpty) return "لا يوجد شيء";

    final parts = <String>[];
    count.forEach((name, c) {
      if (c == 1) {
        parts.add("واحد $name");
      } else if (name == "شخص") {
        parts.add("$c أشخاص");
      } else {
        parts.add("$c $name" + (c > 10 ? "ات" : ""));
      }
    });

    if (parts.length == 1) return "يوجد ${parts[0]}";
    if (parts.length == 2) return "يوجد ${parts[0]} و${parts[1]}";
    return "يوجد ${parts.sublist(0, parts.length - 1).join("، ")} و${parts.last}";
  }

  String _buildEnglishSpeech(Map<String, int> count) {
    if (count.isEmpty) return "Nothing detected";

    final parts = <String>[];
    count.forEach((name, c) {
      final s = c > 1 ? "s" : "";
      parts.add("$c $name$s");
    });

    if (parts.length == 1) return "I see ${parts[0]}";
    if (parts.length == 2) return "I see ${parts[0]} and ${parts[1]}";
    return "I see ${parts.sublist(0, parts.length - 1).join(", ")} and ${parts.last}";
  }

  Future<void> _speak(String text) async {
    if (_isSpeaking) await _tts.stop();
    _isSpeaking = true;
    print("يقول: $text");
    await _tts.speak(text);
    _isSpeaking = false;
  }

  @override
  void dispose() {
    _interpreter.close();
    _tts.stop();
    super.dispose();
  }
}