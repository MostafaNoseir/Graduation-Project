// import 'dart:io';
// import 'package:http/http.dart' as http;
// import 'dart:convert';
//
// class ApiService {
//   final String baseUrl;
//
//   ApiService({required this.baseUrl});
//
//   // دالة رفع صورة
//   Future<String> uploadImage(File imageFile) async {
//     var url = Uri.parse('$baseUrl/predict');
//     var request = http.MultipartRequest('POST', url);
//     request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));
//
//     var streamedResponse = await request.send();
//     var response = await http.Response.fromStream(streamedResponse);
//
//     if (response.statusCode == 200) {
//       var data = json.decode(response.body);
//       return data['prediction'];
//     } else {
//       throw Exception("خطأ في التواصل مع السيرفر");
//     }
//   }
// }
