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

  /// تشغيل/إيقاف الفلاش
  Future<void> toggleFlash() async {
    if (controller == null || !controller!.value.isInitialized) return;
    final next = controller!.value.flashMode == FlashMode.torch
        ? FlashMode.off
        : FlashMode.torch;
    await controller!.setFlashMode(next);
  }

  Future<void> setFlashOff() async {
    if (controller == null || !controller!.value.isInitialized) return;
    await controller!.setFlashMode(FlashMode.off);
  }

  bool get isFlashOn =>
      controller?.value.flashMode == FlashMode.torch;

  void dispose() {
    controller?.dispose();
  }
}