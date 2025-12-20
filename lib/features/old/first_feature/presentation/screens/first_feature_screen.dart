// import 'dart:io';
// import 'package:flutter/material.dart';
// import 'package:image_picker/image_picker.dart';
// import 'package:graduation_project/services/api_service.dart';
//
// final apiService = ApiService(baseUrl: 'http://10.0.2.2:8000');
//
// class FirstFeatureScreen extends StatefulWidget {
//   const FirstFeatureScreen({super.key});
//
//   @override
//   State<FirstFeatureScreen> createState() => _FirstFeatureScreenState();
// }
//
// class _FirstFeatureScreenState extends State<FirstFeatureScreen> {
//   File? imageFile;
//   String prediction = "";
//
//   final picker = ImagePicker();
//
//   // دالة لفتح الكاميرا والتقاط صورة
//   Future takePhoto() async {
//     final pickedFile = await picker.pickImage(source: ImageSource.camera);
//
//     if (pickedFile != null) {
//       setState(() {
//         imageFile = File(pickedFile.path);
//       });
//       sendImage();
//     }
//   }
//
//   // دالة رفع الصورة للـ API
//   void sendImage() async {
//     if (imageFile != null) {
//       try {
//         String result = await apiService.uploadImage(imageFile!);
//         setState(() {
//           prediction = result;
//         });
//       } catch (e) {
//         print(e);
//         setState(() {
//           prediction = "حدث خطأ أثناء التواصل مع السيرفر";
//         });
//       }
//     }
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(title: const Text("اختبار الميزة الأولى")),
//       body: Center(
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             imageFile == null
//                 ? const Text("لم يتم التقاط صورة بعد")
//                 : Image.file(imageFile!, height: 200),
//             const SizedBox(height: 20),
//             Text("Prediction: $prediction"),
//             const SizedBox(height: 20),
//             ElevatedButton(
//               onPressed: takePhoto,
//               child: const Text("التقط صورة بالكاميرا"),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }
