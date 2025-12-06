// lib/main.dart
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'yolo_detector.dart';

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
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

class CameraScreen extends StatefulWidget {
  @override
  _CameraScreenState createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  late YoloDetector _detector;
  bool _isDetecting = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _detector = YoloDetector()..loadModel();
  }

  Future<void> _initCamera() async {
    _controller = CameraController(cameras[0], ResolutionPreset.high);
    await _controller.initialize();
    if (mounted) setState(() {});

    _controller.startImageStream((image) {
      if (!_isDetecting) {
        _isDetecting = true;
        _detector.detectFrame(image).then((_) => _isDetecting = false);
      }
    });
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
          const Align(
            alignment: Alignment.bottomCenter,
            child: Padding(
              padding: EdgeInsets.all(30),
              child: Text(
                "وجه الكاميرا نحو الأشياء",
                style: TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold),
              ),
            ),
          ),
        ],
      ),
      // في نهاية build method في CameraScreen

      bottomNavigationBar: Container(
        color: Colors.black54,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            ElevatedButton.icon(
              onPressed: () async {
                if (!_detector.isArabic) await _detector.toggleLanguage();
              },
              style: ElevatedButton.styleFrom(backgroundColor: _detector.isArabic ? Colors.green : Colors.grey),
              icon: Icon(Icons.volume_up),
              label: Text("العربية", style: TextStyle(fontSize: 18)),
            ),
            ElevatedButton.icon(
              onPressed: () async {
                if (_detector.isArabic) await _detector.toggleLanguage();
              },
              style: ElevatedButton.styleFrom(backgroundColor: !_detector.isArabic ? Colors.blue : Colors.grey),
              icon: Icon(Icons.volume_up),
              label: Text("English", style: TextStyle(fontSize: 18)),
            ),
          ],
        ),
      ),
    );
    // في نهاية build method في CameraScreen

}

  @override
  void dispose() {
    _controller.dispose();
    _detector.dispose();
    super.dispose();
  }

}