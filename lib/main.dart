import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:graduation_project/screens/onboarding_screen.dart';
import 'package:graduation_project/screens/camera_screen.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();

  // ✅ تحقق هل سبق للمستخدم إتمام شاشة الشرح
  final prefs = await SharedPreferences.getInstance();
  final onboardingDone = prefs.getBool('onboarding_done') ?? false;

  runApp(MyApp(showOnboarding: !onboardingDone));
}

class MyApp extends StatelessWidget {
  final bool showOnboarding;
  const MyApp({super.key, required this.showOnboarding});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      // ✅ أول مرة → شاشة الشرح | بعدها مباشرة → الكاميرا
      home: showOnboarding ? const OnboardingScreen() : const CameraScreen(),
    );
  }
}