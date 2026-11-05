# Photo → CNC Relief

تطبيق Python يحول صورة بورتريه إلى:

Image → Depth Map → 3D Relief → STL

## التشغيل

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

في أول تشغيل سيتم تنزيل نموذج:
`Depth-Anything-V2-Small`

## ملاحظات CNC

الـSTL الناتج هو Relief هندسي بقاعدة مسطحة. قبل التصنيع:
1. افتحه في برنامج CAM.
2. تحقق من الـscale بالـmm.
3. تحقق من الوجه والحدود والـbase.
4. أنشئ Roughing وFinishing toolpaths.
5. اختر Ball Nose / V-bit حسب مستوى التفاصيل والخامة.

هذا الإصدار Prototype: أفضل جودة للوجه تحتاج لاحقًا preprocessing للوجه، face-aware depth، وإمكانية تعديل الـdepth يدويًا.
