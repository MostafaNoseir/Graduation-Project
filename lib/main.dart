// lib/main.dart
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'yolo_detector.dart';
import 'face_detector.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:image/image.dart' as imgLib;

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Hive.initFlutter();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'مساعد المكفوفين',
      debugShowCheckedModeBanner: false,
      home: CameraScreen(),
    );
  }
}

enum DetectionMode { objects, faces }

enum FaceSubMode { recognize, enroll }

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late YoloDetector _objectDetector;
  late FaceDetector _faceDetector;

  DetectionMode _mode = DetectionMode.objects;
  FaceSubMode _faceSubMode = FaceSubMode.recognize;

  List<Float32List> _tempEmbeddings = [];
  String _currentPersonName = '';

  @override
  void initState() {
    super.initState();
    _objectDetector = YoloDetector()..loadModel();
    _faceDetector = FaceDetector()..init();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _controller = CameraController(cameras.first, ResolutionPreset.high);
    await _controller.initialize();
    if (mounted) setState(() {});
  }

  Future<void> _captureAndProcess() async {
    try {
      final xFile = await _controller.takePicture();
      final bytes = await xFile.readAsBytes();

      if (_mode == DetectionMode.objects) {
        // موديل الأشياء: استخدم detectFromBytes
        await _objectDetector.detectFromBytes(bytes);
      } else {
        // موديل الوجوه
        final emb = await _faceDetector.getEmbedding(bytes);
        if (emb == null) {
          await _objectDetector.speak("لا يوجد وجه واضح");
          return;
        }

        if (_faceSubMode == FaceSubMode.enroll) {
          _tempEmbeddings.add(emb);
          await _objectDetector.speak("تم التقاط الصورة رقم ${_tempEmbeddings.length} من 5");

          if (_tempEmbeddings.length >= 5) {
            String name = '';
            await showDialog(
              context: context,
              builder: (ctx) => AlertDialog(
                title: Text("أدخل اسم الشخص"),
                content: TextField(onChanged: (v) => name = v),
                actions: [
                  TextButton(
                    onPressed: () async {
                      if (name.isNotEmpty) {
                        await _faceDetector.enroll(name, _tempEmbeddings);
                        await _objectDetector.speak("تم تسجيل $name بنجاح");
                        _tempEmbeddings.clear();
                        Navigator.pop(ctx);
                      }
                    },
                    child: Text("حفظ"),
                  ),
                ],
              ),
            );
          }
        } else {
          final name = await _faceDetector.recognize(emb);
          await _objectDetector.speak("الوجه هو $name");
        }
      }
    } catch (e) {
      print("خطأ في التقاط الصورة: $e");
      await _objectDetector.speak("حدث خطأ");
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          CameraPreview(_controller),
          // زر اللغة أعلى يمين
          Align(
            alignment: Alignment.topRight,
            child: Padding(
              padding: EdgeInsets.all(16),
              child: IconButton(
                icon: Icon(Icons.language, color: Colors.white, size: 36),
                onPressed: _objectDetector.toggleLanguage,
              ),
            ),
          ),
          // زرارين للوجوه فقط (يسار)
          if (_mode == DetectionMode.faces)
            Align(
              alignment: Alignment.centerLeft,
              child: Padding(
                padding: EdgeInsets.all(20),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    ElevatedButton(
                      onPressed: () => setState(() => _faceSubMode = FaceSubMode.enroll),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _faceSubMode == FaceSubMode.enroll ? Colors.green : Colors.grey,
                      ),
                      child: Text("تسجيل جديد"),
                    ),
                    SizedBox(height: 20),
                    ElevatedButton(
                      onPressed: () => setState(() => _faceSubMode = FaceSubMode.recognize),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _faceSubMode == FaceSubMode.recognize ? Colors.blue : Colors.grey,
                      ),
                      child: Text("تعرف"),
                    ),
                  ],
                ),
              ),
            ),
          // زر التقاط صورة
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: EdgeInsets.only(bottom: 120),
              child: FloatingActionButton(
                onPressed: _captureAndProcess,
                backgroundColor: Colors.white,
                child: Icon(Icons.camera_alt, size: 40, color: Colors.black),
              ),
            ),
          ),
          // دواير اختيار الموديل
          Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: EdgeInsets.only(bottom: 20),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _buildModeButton(DetectionMode.objects, Icons.visibility, "أشياء"),
                  SizedBox(width: 60),
                  _buildModeButton(DetectionMode.faces, Icons.face, "وجوه"),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildModeButton(DetectionMode mode, IconData icon, String label) {
    final isSelected = _mode == mode;
    return GestureDetector(
      onTap: () {
        setState(() => _mode = mode);
        _objectDetector.speak("تم تفعيل $label");
      },
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircleAvatar(
            radius: 35,
            backgroundColor: isSelected ? Colors.white : Colors.white.withOpacity(0.3),
            child: Icon(icon, size: 40, color: isSelected ? Colors.black : Colors.white),
          ),
          SizedBox(height: 8),
          Text(label, style: TextStyle(color: Colors.white, fontSize: 16)),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    _objectDetector.dispose();
    _faceDetector.dispose();
    super.dispose();
  }
}