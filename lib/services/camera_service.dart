import 'package:camera/camera.dart';

class CameraService {
  CameraController? controller;

  Future<void> init(CameraDescription camera) async {
    controller = CameraController(
      camera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    await controller!.initialize();
  }

  /// 📸 تصوير صورة واحدة
  Future<XFile?> takePicture() async {
    if (controller == null || !controller!.value.isInitialized) return null;
    if (controller!.value.isTakingPicture) return null;

    return await controller!.takePicture();
  }

  void dispose() {
    controller?.dispose();
  }
}
