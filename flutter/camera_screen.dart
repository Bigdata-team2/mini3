import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/image_service.dart';

/// 카메라/갤러리에서 이미지를 선택하고
/// FastAPI로 업로드해 재료 인식 및 레시피 추천까지 진행하는 화면
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  final ImagePicker _picker = ImagePicker();
  final ImageService _imageService = ImageService();

  File? _selectedImage;
  bool _isLoading = false;

  List<String> _ingredients = [];
  List<Map<String, dynamic>> _recommendations = [];

  Future<void> _pickFromCamera() async {
    try {
      final XFile? picked = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 95,
      );
      if (picked != null) {
        setState(() {
          _selectedImage = File(picked.path);
          _ingredients = [];
          _recommendations = [];
        });
      }
    } catch (e) {
      debugPrint("⚠️ 카메라 선택 오류: $e");
    }
  }

  Future<void> _pickFromGallery() async {
    try {
      final XFile? picked = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 95,
      );
      if (picked != null) {
        setState(() {
          _selectedImage = File(picked.path);
          _ingredients = [];
          _recommendations = [];
        });
      }
    } catch (e) {
      debugPrint("⚠️ 갤러리 선택 오류: $e");
    }
  }

  /// 인식만 (POST /predict)
  Future<void> _uploadToFastAPI() async {
    if (_selectedImage == null) return;
    setState(() => _isLoading = true);

    try {
      final result = await _imageService.uploadAndPredict(_selectedImage!);
      setState(() {
        _ingredients = result;
        _recommendations = [];
      });
    } catch (e) {
      debugPrint("⚠️ 업로드/인식 오류: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("인식 중 오류가 발생했습니다.")),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  /// 인식 + 추천 (POST /predict-and-recommend)
  Future<void> _uploadAndRecommend() async {
    if (_selectedImage == null) return;
    setState(() => _isLoading = true);

    try {
      final data =
          await _imageService.uploadAndRecommend(_selectedImage!, topK: 5);

      final ingredients =
          (data['ingredients'] is List) ? List<String>.from(data['ingredients']) : <String>[];

      final recsRaw = (data['recommendations'] is List)
          ? List<Map<String, dynamic>>.from(
              (data['recommendations'] as List).map((e) => Map<String, dynamic>.from(e)))
          : <Map<String, dynamic>>[];

      setState(() {
        _ingredients = ingredients;
        _recommendations = recsRaw;
      });
    } catch (e) {
      debugPrint("⚠️ 업로드/추천 오류: $e");
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("추천 중 오류가 발생했습니다.")),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Widget _buildImagePreview() {
    if (_selectedImage == null) {
      return Container(
        height: 220,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: Colors.grey.shade200,
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Text("이미지를 선택하세요"),
      );
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Image.file(
        _selectedImage!,
        height: 220,
        width: double.infinity,
        fit: BoxFit.cover,
      ),
    );
  }

  Widget _buildIngredients() {
    if (_ingredients.isEmpty) {
      return const Text("인식된 재료가 없습니다.");
    }
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: _ingredients
          .map((ing) => Chip(
                label: Text(ing),
                backgroundColor: Colors.green.shade50,
              ))
          .toList(),
    );
  }

  Widget _buildRecommendations() {
    if (_recommendations.isEmpty) {
      return const Text("추천 결과가 없습니다.");
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        for (final rec in _recommendations)
          Card(
            child: ListTile(
              title: Text(rec['metadata']?['title']?.toString() ?? '제목 없음'),
              subtitle: Text(
                rec['metadata']?['ingredients']?.toString() ??
                    rec['document']?.toString() ??
                    '상세 없음',
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              trailing: Text(
                rec['distance'] != null
                    ? (rec['distance'] as num).toStringAsFixed(3)
                    : '',
                style: const TextStyle(fontFeatures: [FontFeature.tabularFigures()]),
              ),
            ),
          ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("재료 인식 & 레시피 추천")),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _buildImagePreview(),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _pickFromCamera,
                    icon: const Icon(Icons.photo_camera),
                    label: const Text("카메라"),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _pickFromGallery,
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text("갤러리"),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: FilledButton(
                    onPressed: _isLoading ? null : _uploadToFastAPI,
                    child: _isLoading
                        ? const Padding(
                            padding: EdgeInsets.symmetric(vertical: 8.0),
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Text("인식만 (/predict)"),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.tonal(
                    onPressed: _isLoading ? null : _uploadAndRecommend,
                    child: _isLoading
                        ? const Padding(
                            padding: EdgeInsets.symmetric(vertical: 8.0),
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Text("인식+추천 (/predict-and-recommend)"),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),
            const Text("🧾 인식된 재료", style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            _buildIngredients(),
            const SizedBox(height: 24),
            const Text("🍳 추천 레시피", style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            _buildRecommendations(),
          ],
        ),
      ),
    );
  }
}
