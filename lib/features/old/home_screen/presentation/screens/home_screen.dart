// import 'package:flutter/material.dart';
// import 'package:graduation_project/core/core.dart';
// import 'package:graduation_project/features/first_feature/presentation/screens/first_feature_screen.dart';
//
// class HomeScreen extends StatelessWidget {
//   const HomeScreen({super.key});
//
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       body: Container(
//         decoration: BoxDecoration(
//           image: DecorationImage(
//             image: AssetImage('assets/test_bg.jpg'),
//             fit: BoxFit.cover,
//           ),
//         ),
//         child: Padding(
//           padding: EdgeInsets.symmetric(vertical:AppSizes.h300),
//           child: Align(
//             alignment: Alignment.bottomCenter,
//             child: CustomElevatedButton(
//               height: AppSizes.h130,
//               width: AppSizes.w800,
//               text: 'First Feature Test',
//               textStyle: TextStyle(
//                 fontWeight: FontWeight.bold,
//                 fontSize: AppSizes.fontS70,
//                 color: AppColors.primaryBlue,
//               ),
//               backgroundColor: Colors.white,
//               borderRadius: AppSizes.r100,
//               onPressed: () {
//                 Navigator.push(
//                   context,
//                   MaterialPageRoute(builder: (context) => FirstFeatureScreen()),
//                 );
//               },
//             ),
//           ),
//         ),
//       ),
//     );
//   }
// }
