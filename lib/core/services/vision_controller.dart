// // lib/core/services/vision_controller.dart
// import 'dart:typed_data';
// import 'package:camera/camera.dart';
//
// import 'yolo_detector.dart';
// import 'text_detector.dart';
// import 'color_detector.dart';
// import 'ttf_service.dart';
//
// enum VisionMode { objects, text, color }
//
// class VisionController {
//   VisionMode _mode = VisionMode.objects;
//
//   final YoloDetector yolo;
//   final TextDetector textDetector;
//   final ColorDetector colorDetector;
//   final TtsService tts;
//
//   VisionController({
//     required this.yolo,
//     required this.textDetector,
//     required this.colorDetector,
//     required this.tts,
//   });
//
//   VisionMode get mode => _mode;
//
//   Future<void> switchMode(VisionMode newMode) async {
//     _mode = newMode;
//
//     switch (newMode) {
//       case VisionMode.objects:
//         await tts.speak(
//             tts.isArabic ? "تم تفعيل كشف الأجسام" : "Object detection activated");
//         break;
//
//       case VisionMode.text:
//         await tts.speak(
//             tts.isArabic ? "تم تفعيل قراءة النص" : "Text reading activated");
//         break;
//
//       case VisionMode.color:
//         await tts.speak(
//             tts.isArabic ? "تم تفعيل كشف الألوان" : "Color detection activated");
//         break;
//     }
//   }
//
//   // 📷 معالجة الفريم حسب المود
//   Future<void> processFrame(CameraImage image) async {
//     if (_mode == VisionMode.objects) {
//       await yolo.detectFrame(image);
//     }
//   }
//
//   // 📸 صورة ثابتة (نص / لون)
//   Future<void> processImage({
//     String? imagePath,
//     Uint8List? bytes,
//   }) async {
//     if (_mode == VisionMode.text && imagePath != null) {
//       final text = await textDetector.extractText(imagePath);
//       if (text != null) await tts.speak(text);
//     }
//
//     if (_mode == VisionMode.color && bytes != null) {
//       final color = await colorDetector.detectDominantColor(bytes);
//       if (color != null) await tts.speak(color);
//     }
//   }
// }
