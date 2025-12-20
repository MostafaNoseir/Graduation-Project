// import 'dart:async';
// import 'dart:math' as math;
// import 'dart:typed_data';
// import 'package:flutter/material.dart';
// import 'package:flutter/services.dart';
// import 'package:camera/camera.dart';
// import 'package:tflite_flutter/tflite_flutter.dart';
// import 'package:flutter_tts/flutter_tts.dart';
//
// late List<CameraDescription> cameras;
//
// void main() async {
//   WidgetsFlutterBinding.ensureInitialized();
//   cameras = await availableCameras();
//   runApp(const MyApp());
// }
//
// class MyApp extends StatelessWidget {
//   const MyApp({super.key});
//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       title: 'مساعد المكفوفين',
//       debugShowCheckedModeBanner: false,
//       theme: ThemeData.dark(),
//       home: ObjectDetectionScreen(camera: cameras.first),
//     );
//   }
// }
//
// class ObjectDetectionScreen extends StatefulWidget {
//   final CameraDescription camera;
//   const ObjectDetectionScreen({required this.camera, super.key});
//
//   @override
//   State<ObjectDetectionScreen> createState() => _ObjectDetectionScreenState();
// }
//
// class _ObjectDetectionScreenState extends State<ObjectDetectionScreen> {
//   CameraController? _controller;           // خليناه nullable بدل late
//   Interpreter? _interpreter;
//
//   List<String> _arabicLabels = [];
//   List<String> _englishLabels = [];
//   String _currentLanguage = 'ar';
//
//   bool _isDetecting = false;
//   final FlutterTts _tts = FlutterTts();
//
//   String _lastSpoken = "";
//   DateTime _lastSpokeTime = DateTime.now().subtract(const Duration(seconds: 10));
//
//   bool _isLoading = true;  // عشان نعرض شاشة تحميل لحد ما كل حاجة تجهز
//
//   @override
//   void initState() {
//     super.initState();
//     _initEverything();
//   }
//
//   Future<void> _initEverything() async {
//     _detectLanguage();
//     await _initTts();
//     await _loadModel();
//     await _loadLabels();
//     await _initCamera();
//
//     if (mounted) {
//       setState(() {
//         _isLoading = false;
//       });
//     }
//   }
//
//   void _detectLanguage() {
//     final locale = WidgetsBinding.instance.platformDispatcher.locale;
//     _currentLanguage = (locale.languageCode == 'ar') ? 'ar' : 'en';
//   }
//
//   Future<void> _initTts() async {
//     await _tts.setLanguage(_currentLanguage == 'ar' ? "ar-XA" : "en-US");
//     await _tts.setSpeechRate(0.5);
//   }
//
//   Future<void> _loadModel() async {
//     try {
//       _interpreter = await Interpreter.fromAsset('assets/models/yolov8n_float32.tflite');
//       print("تم تحميل النموذج");
//     } catch (e) {
//       print("خطأ تحميل النموذج: $e");
//     }
//   }
//
//   Future<void> _loadLabels() async {
//     try {
//       final arabicData = await rootBundle.loadString('assets/labels/labels_arabic.txt');
//       final englishData = await rootBundle.loadString('assets/labels/labels.txt');
//
//       _arabicLabels = arabicData.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();
//       _englishLabels = englishData.split('\n').map((e) => e.trim()).where((e) => e.isNotEmpty).toList();
//     } catch (e) {
//       print("خطأ تحميل الـ labels: $e");
//     }
//   }
//
//   Future<void> _initCamera() async {
//     _controller = CameraController(
//       widget.camera,
//       ResolutionPreset.high,
//       enableAudio: false,
//       imageFormatGroup: ImageFormatGroup.bgra8888,
//     );
//
//     try {
//       await _controller!.initialize();
//       await _controller!.startImageStream((image) async {
//         if (_isDetecting || _interpreter == null || _arabicLabels.isEmpty) return;
//         _isDetecting = true;
//         try {
//           await _processFrame(image);
//         } catch (e) {
//           debugPrint("خطأ: $e");
//         } finally {
//           _isDetecting = false;
//         }
//       });
//     } catch (e) {
//       print("خطأ في الكاميرا: $e");
//     }
//   }
//
//   Future<void> _processFrame(CameraImage image) async {
//     final input = _preprocessImage(image);
//
//     var output = <List<List<double>>>[
//       List.generate(84, (_) => List<double>.filled(8400, 0.0))
//     ];
//
//     _interpreter!.run(input, output);
//     final result = output[0];
//
//     List<Map<String, dynamic>> detections = [];
//
//     for (int i = 0; i < 8400; i++) {
//       final confidence = result[4][i];
//       if (confidence < 0.5) continue;
//
//       double maxScore =  0.0;
//     int maxId = 0;
//     for (int c = 4; c < 84; c++) {
//     if (result[c][i] > maxScore) {
//     maxScore = result[c][i];
//     maxId = c - 4;
//     }
//     }
//
//     if (maxId < (_currentLanguage == 'ar' ? _arabicLabels.length : _englishLabels.length)) {
//     final label = _currentLanguage == 'ar' ? _arabicLabels[maxId] : _englishLabels[maxId];
//     detections.add({'label': label, 'confidence': confidence});
//     }
//     }
//
//     if (detections.isNotEmpty) {
//     detections.sort((a, b) => b['confidence'].compareTo(a['confidence']));
//     final topLabel = detections.first['label'];
//     final now = DateTime.now();
//
//     if (topLabel != _lastSpoken || now.difference(_lastSpokeTime).inSeconds > 3) {
//     _lastSpoken = topLabel;
//     _lastSpokeTime = now;
//     final phrase = _currentLanguage == 'ar' ? "$topLabel أمامك" : "$topLabel in front of you";
//     await _tts.speak(phrase);
//     }
//     }
//   }
//
//   Float32List _preprocessImage(CameraImage image) {
//     const int inputSize = 640;
//     final int width = image.width;
//     final int height = image.height;
//     final Uint8List bytes = image.planes[0].bytes;
//
//     final Float32List input = Float32List(1 * 3 * inputSize * inputSize);
//     int index = 0;
//
//     for (int y = 0; y < inputSize; y++) {
//       for (int x = 0; x < inputSize; x++) {
//         final int srcX = (x * width / inputSize).floor();
//         final int srcY = (y * height / inputSize).floor();
//         final int byteIndex = srcY * width + srcX;
//         final double pixel = bytes[byteIndex] / 255.0;
//
//         input[index++] = pixel;
//         input[index++] = pixel;
//         input[index++] = pixel;
//       }
//     }
//     return input;
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     if (_isLoading || _controller == null || !_controller!.value.isInitialized) {
//       return const Scaffold(
//         backgroundColor: Colors.black,
//         body: Center(
//           child: Column(
//             mainAxisAlignment: MainAxisAlignment.center,
//             children: [
//               CircularProgressIndicator(color: Colors.white),
//               SizedBox(height: 20),
//               Text("جاري تحميل النموذج...", style: TextStyle(color: Colors.white, fontSize: 20)),
//             ],
//           ),
//         ),
//       );
//     }
//
//     return Scaffold(
//       backgroundColor: Colors.black,
//       body: Stack(
//         children: [
//           CameraPreview(_controller!),
//           const Align(
//             alignment: Alignment.bottomCenter,
//             child: Padding(
//               padding: EdgeInsets.all(20),
//               child: Text(
//                 "وجه الكاميرا للأمام\nالتطبيق يعمل الآن",
//                 textAlign: TextAlign.center,
//                 style: TextStyle(color: Colors.white, fontSize: 28, fontWeight: FontWeight.bold),
//               ),
//             ),
//           ),
//         ],
//       ),
//     );
//   }
//
//   @override
//   void dispose() {
//     _controller?.dispose();
//     _interpreter?.close();
//     _tts.stop();
//     super.dispose();
//   }
// }
//
// // import 'package:flutter/material.dart';
// // import 'package:flutter_screenutil/flutter_screenutil.dart';
// // import 'package:graduation_project/features/home_screen/presentation/screens/home_screen.dart';
// //
// // void main() {
// //   runApp(const MyApp());
// // }
// //
// // class MyApp extends StatelessWidget {
// //   const MyApp({super.key});
// //
// //   @override
// //   Widget build(BuildContext context) {
// //     return ScreenUtilInit(
// //       designSize: const Size(1080, 1920),
// //       minTextAdapt: true,
// //       splitScreenMode: true,
// //       builder: (context, child) {
// //         return MaterialApp(
// //           title: 'Flutter Demo',
// //           debugShowCheckedModeBanner: false,
// //           home: child,
// //         );
// //       },
// //       child: const HomeScreen(),
// //     );
// //   }
// // }
