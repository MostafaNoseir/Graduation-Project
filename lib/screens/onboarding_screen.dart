// lib/screens/onboarding_screen.dart

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:graduation_project/screens/camera_screen.dart';

const _kOnboardingDone = 'onboarding_done';

class OnboardingScreen extends StatefulWidget {
  /// إذا فُتحت من داخل التطبيق لا نحتاج لحفظ الـ flag مجدداً
  final bool fromApp;
  const OnboardingScreen({super.key, this.fromApp = false});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final FlutterTts _tts = FlutterTts();
  final PageController _pageController = PageController();

  int _currentPage = 0;
  bool _disposed = false;

  // ══════════════════════════════════════════════════════════════
  //  محتوى الشرائح — عربي
  // ══════════════════════════════════════════════════════════════
  static const List<_SlideData> _slidesAr = [
    _SlideData(
      icon: Icons.visibility,
      title: "مرحباً بك في آفا",
      body:
      "آفا، اختصار لـ \"المساعد البصري الذكي\" بالإنجليزية AI Vision Assistant، "
          "هو تطبيقك الذكي المصمم لمساعدة ضعاف البصر والمكفوفين على فهم ما حولهم.\n\n"
          "اضغط على الشاشة للانتقال للخطوة التالية.",
    ),
    _SlideData(
      icon: Icons.touch_app,
      title: "طريقة الاستخدام الأساسية",
      body:
      "• اضغط ضغطة واحدة على زر العين الموجود في منتصف الشاشة، فوق أزرار الأوضاع السفلية مباشرةً، لالتقاط الصورة وتحليلها.\n\n"
          "• اضغط مرتين بسرعة في أي مكان للتبديل بين الكاميرا الأمامية والخلفية.\n\n"
          "• اضغط مطولاً في أي مكان لتفعيل الأوامر الصوتية.",
    ),
    _SlideData(
      icon: Icons.category,
      title: "أوضاع الكشف",
      body:
      "في أسفل الشاشة توجد خمسة أوضاع مرتبة من اليسار لليمين:\n\n"
          "أشياء — يتعرف على الأشياء من حولك.\n"
          "ألوان — يكشف اللون الرئيسي في الصورة.\n"
          "نص — يقرأ النصوص المكتوبة.\n"
          "عملة — يتعرف على العملات المصرية.\n"
          "وجه — يتعرف على الأشخاص المسجّلين.",
    ),
    _SlideData(
      icon: Icons.mic,
      title: "الأوامر الصوتية",
      body:
      "اضغط مطولاً ثم قل أحد هذه الأوامر:\n\n"
          "«شوف» أو «أشياء» — لكشف الأشياء.\n"
          "«لون» أو «ألوان» — لكشف اللون.\n"
          "«اقرأ» أو «نص» — لقراءة النص.\n"
          "«فلوس» أو «نقود» — للتعرف على العملة.\n"
          "«وجه» أو «مين» — للتعرف على الوجه.\n"
          "«فلاش» أو «كشاف» — لتشغيل أو إيقاف الفلاش.\n"
          "«انجليزي» أو «english» — لتغيير اللغة للإنجليزية.\n"
          "«arabic» — لتغيير اللغة للعربية.\n"
          "«أمامي» أو «قدام» — للكاميرا الأمامية.\n"
          "«خلفي» أو «ورا» — للكاميرا الخلفية.\n"
          "«تعليمات» أو «مساعدة» — لإعادة عرض هذا الشرح.",
    ),
    _SlideData(
      icon: Icons.face,
      title: "وضع التعرف على الوجه",
      body:
      "في وضع الوجه، ستجد ثلاثة أزرار في منتصف الشاشة فوق أزرار الأوضاع مباشرةً، مرتبة من اليسار لليمين:\n\n"
          "الزر الأبيض — التعرف على الشخص أمامك.\n"
          "الزر الأزرق — إضافة وجه جديد، سيطلب منك كتابة الاسم.\n"
          "الزر الأحمر — حذف جميع الوجوه المحفوظة.",
    ),
    _SlideData(
      icon: Icons.language,
      title: "تغيير اللغة والفلاش",
      body:
      "اضغط على أيقونة اللغة في أعلى يمين الشاشة للتبديل بين العربية والإنجليزية.\n\n"
          "اضغط على أيقونة الفلاش في أعلى وسط الشاشة لتشغيل أو إيقاف الإضاءة، مفيد في الأماكن المظلمة.\n\n"
          "يمكنك أيضاً قول «english» أو «arabic» لتغيير اللغة، و«فلاش» أو «كشاف» لتشغيل الإضاءة عبر الأوامر الصوتية.",
    ),
    _SlideData(
      icon: Icons.security,
      title: "الأذونات المطلوبة",
      body:
      "لكي يعمل التطبيق بشكل صحيح، سيطلب منك إذنين:\n\n"
          "الكاميرا — لالتقاط الصور وتحليلها.\n"
          "الميكروفون — لتلقي الأوامر الصوتية.\n\n"
          "يرجى الموافقة على كلا الإذنين عند ظهور طلب المنح، وإلا لن يتمكن التطبيق من العمل.",
    ),
    _SlideData(
      icon: Icons.check_circle,
      title: "أنت جاهز!",
      body:
      "التطبيق يتذكر آخر وضع واللغة والكاميرا التي استخدمتها.\n\n"
          "يمكنك في أي وقت إعادة هذا الشرح بالضغط المطول وقول «تعليمات».\n\n"
          "اضغط على الشاشة للبدء في استخدام آفا.",
    ),
  ];

  // ══════════════════════════════════════════════════════════════
  //  محتوى الشرائح — إنجليزي
  // ══════════════════════════════════════════════════════════════
  static const List<_SlideData> _slidesEn = [
    _SlideData(
      icon: Icons.visibility,
      title: "Welcome to AVA",
      body:
      "AVA stands for AI Vision Assistant — your smart visual companion "
          "designed to help the visually impaired understand the world around them.\n\n"
          "Tap the screen to go to the next step.",
    ),
    _SlideData(
      icon: Icons.touch_app,
      title: "Basic Usage",
      body:
      "• Tap the eye button located in the center of the screen, directly above the mode buttons at the bottom, to capture and analyze.\n\n"
          "• Double tap anywhere to switch between front and back camera.\n\n"
          "• Long press anywhere to activate voice commands.",
    ),
    _SlideData(
      icon: Icons.category,
      title: "Detection Modes",
      body:
      "At the bottom of the screen there are five modes ordered from left to right:\n\n"
          "Objects — Identifies objects around you.\n"
          "Colors — Detects the dominant color.\n"
          "Text — Reads written text.\n"
          "Money — Recognizes Egyptian currency.\n"
          "Face — Recognizes enrolled people.",
    ),
    _SlideData(
      icon: Icons.mic,
      title: "Voice Commands",
      body:
      "Long press then say one of these commands:\n\n"
          "\"object\" or \"what\" — to detect objects.\n"
          "\"color\" — to detect color.\n"
          "\"read\" or \"text\" — to read text.\n"
          "\"money\" or \"currency\" — to detect currency.\n"
          "\"face\" or \"who\" — to recognize face.\n"
          "\"flash\" or \"torch\" — to toggle the flashlight.\n"
          "\"arabic\" — to switch to Arabic.\n"
          "\"english\" — to switch to English.\n"
          "\"selfie\" or \"forward\" — to switch to front camera.\n"
          "\"back\" or \"backward\" — to switch to back camera.\n"
          "\"instructions\" or \"help\" — to replay this guide.",
    ),
    _SlideData(
      icon: Icons.face,
      title: "Face Recognition Mode",
      body:
      "In face mode, you will find three buttons in the center of the screen directly above the mode buttons, ordered from left to right:\n\n"
          "White button — Recognize the person in front of you.\n"
          "Blue button — Add a new face, you will be asked to type a name.\n"
          "Red button — Delete all saved faces.",
    ),
    _SlideData(
      icon: Icons.language,
      title: "Language & Flash",
      body:
      "Tap the language icon at the top right to switch between Arabic and English.\n\n"
          "Tap the flash icon at the top center to toggle the flashlight — useful in dark environments.\n\n"
          "You can also say \"arabic\" or \"english\" to change language, and \"flash\" or \"torch\" to toggle the light via voice commands.",
    ),
    _SlideData(
      icon: Icons.security,
      title: "Required Permissions",
      body:
      "For the app to work correctly, it will request two permissions:\n\n"
          "Camera — To capture and analyze images.\n"
          "Microphone — To receive voice commands.\n\n"
          "Please grant both permissions when prompted, otherwise the app will not function properly.",
    ),
    _SlideData(
      icon: Icons.check_circle,
      title: "You're Ready!",
      body:
      "The app remembers your last mode, language, and camera setting.\n\n"
          "You can replay this guide at any time by long pressing and saying \"instructions\".\n\n"
          "Tap the screen to start using AVA.",
    ),
  ];

  // ══════════════════════════════════════════════════════════════
  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    _initTts();
  }

  Future<void> _initTts() async {
    await _tts.setSpeechRate(0.42);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);
    await Future.delayed(const Duration(milliseconds: 600));
    if (!_disposed) _loopCurrentSlide();
  }

  // ── قراءة الشريحة بالعربي ثم الإنجليزي بشكل متكرر ─────────────────────
  Future<void> _loopCurrentSlide() async {
    final page = _currentPage;

    while (!_disposed && _currentPage == page) {
      // ── عربي ──
      if (_disposed || _currentPage != page) break;
      await _tts.setLanguage("ar-EG");
      final arSlide = _slidesAr[page];
      final arText = _clean("${arSlide.title}. ${arSlide.body}");
      await _speakAndWait(arText);

      if (_disposed || _currentPage != page) break;
      await Future.delayed(const Duration(milliseconds: 800));

      // ── إنجليزي ──
      if (_disposed || _currentPage != page) break;
      await _tts.setLanguage("en-US");
      final enSlide = _slidesEn[page];
      final enText = _clean("${enSlide.title}. ${enSlide.body}");
      await _speakAndWait(enText);

      if (_disposed || _currentPage != page) break;
      await Future.delayed(const Duration(milliseconds: 1200));
    }
  }

  String _clean(String text) =>
      text.replaceAll(RegExp(r'\s+'), ' ').trim();

  // ── ينطق النص وينتظر حتى ينتهي فعلاً ──────────────────────────────────
  Future<void> _speakAndWait(String text) async {
    final completer = Completer<void>();
    _tts.setCompletionHandler(() {
      if (!completer.isCompleted) completer.complete();
    });
    _tts.setCancelHandler(() {
      if (!completer.isCompleted) completer.complete();
    });
    await _tts.speak(text);
    await completer.future.timeout(
      const Duration(seconds: 60),
      onTimeout: () {},
    );
    _tts.setCompletionHandler(() {});
    _tts.setCancelHandler(() {});
  }

  void _nextSlide() {
    if (_currentPage < _slidesAr.length - 1) {
      _tts.stop();
      _pageController.nextPage(
        duration: const Duration(milliseconds: 350),
        curve: Curves.easeInOut,
      );
    } else {
      _finish();
    }
  }

  Future<void> _finish() async {
    _disposed = true;
    await _tts.stop();
    // حفظ الـ flag فقط عند الفتح الأول (وليس عند فتحها من داخل التطبيق)
    if (!widget.fromApp) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_kOnboardingDone, true);
    }
    if (!mounted) return;
    if (widget.fromApp) {
      // العودة للشاشة السابقة فقط
      Navigator.of(context).pop();
    } else {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const CameraScreen()),
      );
    }
  }

  @override
  void dispose() {
    _disposed = true;
    _tts.stop();
    _pageController.dispose();
    super.dispose();
  }

  // ══════════════════════════════════════════════════════════════
  @override
  Widget build(BuildContext context) {
    final mq = MediaQuery.of(context);
    final sw = mq.size.width;
    final sh = mq.size.height;

    // نعرض الشريحة بالعربية دائماً (اللغتان تُقرآن، النص المعروض عربي)
    final slides = _slidesAr;

    return Scaffold(
      backgroundColor: const Color(0xFF0A0A0A),
      body: GestureDetector(
        onTap: _nextSlide,
        behavior: HitTestBehavior.opaque,
        child: SafeArea(
          child: Stack(
            children: [
              // ── الصفحات ───────────────────────────────────────
              PageView.builder(
                controller: _pageController,
                physics: const NeverScrollableScrollPhysics(),
                onPageChanged: (i) {
                  setState(() => _currentPage = i);
                  _loopCurrentSlide();
                },
                itemCount: slides.length,
                itemBuilder: (_, i) => _buildSlide(i, sw, sh),
              ),

              // ── مؤشر الصفحات ──────────────────────────────────
              Positioned(
                bottom: sh * 0.035,
                left: 0,
                right: 0,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(
                    slides.length,
                        (i) => AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      margin: const EdgeInsets.symmetric(horizontal: 4),
                      width: i == _currentPage ? 22 : 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: i == _currentPage
                            ? Colors.white
                            : Colors.white38,
                        borderRadius: BorderRadius.circular(4),
                      ),
                    ),
                  ),
                ),
              ),

              // ── تلميح الضغط ───────────────────────────────────
              Positioned(
                bottom: sh * 0.075,
                left: 0,
                right: 0,
                child: Center(
                  child: Text(
                    _currentPage == slides.length - 1
                        ? "ابدأ الآن / Start Now"
                        : "اضغط للمتابعة / Tap to continue",
                    style: TextStyle(
                      color: Colors.white54,
                      fontSize: (sw * 0.030).clamp(10.0, 14.0),
                      letterSpacing: 0.5,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSlide(int index, double sw, double sh) {
    final arSlide = _slidesAr[index];
    final enSlide = _slidesEn[index];

    return SingleChildScrollView(
      padding: EdgeInsets.symmetric(
          horizontal: sw * 0.07, vertical: sh * 0.04),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(height: sh * 0.02),

          // أيقونة
          Container(
            width: sw * 0.25,
            height: sw * 0.25,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.white10,
              border: Border.all(color: Colors.white24, width: 1.5),
            ),
            child:
            Icon(arSlide.icon, size: sw * 0.12, color: Colors.white),
          ),

          SizedBox(height: sh * 0.03),

          // ── عنوان عربي ──
          Text(
            arSlide.title,
            textAlign: TextAlign.center,
            textDirection: TextDirection.rtl,
            style: TextStyle(
              color: Colors.white,
              fontSize: (sw * 0.055).clamp(17.0, 26.0),
              fontWeight: FontWeight.bold,
              height: 1.3,
            ),
          ),

          SizedBox(height: sh * 0.008),

          // ── عنوان إنجليزي ──
          Text(
            enSlide.title,
            textAlign: TextAlign.center,
            textDirection: TextDirection.ltr,
            style: TextStyle(
              color: Colors.white54,
              fontSize: (sw * 0.040).clamp(13.0, 20.0),
              fontWeight: FontWeight.w500,
              height: 1.3,
            ),
          ),

          SizedBox(height: sh * 0.025),

          // ── محتوى عربي ──
          _contentBox(
            text: arSlide.body,
            isArabic: true,
            sw: sw,
          ),

          SizedBox(height: sh * 0.015),

          // ── فاصل ──
          Row(
            children: [
              Expanded(child: Divider(color: Colors.white24)),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 10),
                child: Text("EN",
                    style: TextStyle(
                        color: Colors.white38,
                        fontSize: (sw * 0.03).clamp(10.0, 13.0))),
              ),
              Expanded(child: Divider(color: Colors.white24)),
            ],
          ),

          SizedBox(height: sh * 0.015),

          // ── محتوى إنجليزي ──
          _contentBox(
            text: enSlide.body,
            isArabic: false,
            sw: sw,
          ),

          SizedBox(height: sh * 0.10),
        ],
      ),
    );
  }

  Widget _contentBox(
      {required String text, required bool isArabic, required double sw}) {
    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(sw * 0.045),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.1)),
      ),
      child: Text(
        text,
        textAlign: isArabic ? TextAlign.right : TextAlign.left,
        textDirection: isArabic ? TextDirection.rtl : TextDirection.ltr,
        style: TextStyle(
          color: Colors.white70,
          fontSize: (sw * 0.034).clamp(11.0, 15.0),
          height: 1.7,
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════
class _SlideData {
  final IconData icon;
  final String title;
  final String body;
  const _SlideData(
      {required this.icon, required this.title, required this.body});
}