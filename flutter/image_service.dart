import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

/// FastAPI 서버와 통신하는 이미지 업로드/예측 서비스
class ImageService {
  /// ✅ 서버 IP만 바꿔주세요!
  final String fastApiPredictUrl = "http://<SERVER_IP>:8081/predict";
  final String fastApiPredictAndRecommendUrl =
      "http://<SERVER_IP>:8081/predict-and-recommend";

  /// POST /predict : 이미지 업로드 → 재료 인식
  Future<List<String>> uploadAndPredict(File imageFile) async {
    try {
      final uri = Uri.parse(fastApiPredictUrl);
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('image', imageFile.path));

      final resp = await request.send();

      if (resp.statusCode != 200) {
        final errBody = await resp.stream.bytesToString();
        throw HttpException(
          "FastAPI /predict error: ${resp.statusCode} $errBody",
        );
      }

      final respStr = await resp.stream.bytesToString();
      final data = jsonDecode(respStr);

      if (data is Map && data['ingredients'] != null) {
        return List<String>.from(data['ingredients']);
      }
      if (data is List) {
        return List<String>.from(data);
      }
      return [];
    } catch (e) {
      // 로깅만 하고 빈 리스트 반환
      print("⚠️ FastAPI 연결 에러 (/predict): $e");
      return [];
    }
  }

  /// POST /predict-and-recommend : 이미지 업로드 → 재료 인식 → 레시피 추천(원샷)
  /// 반환 예시:
  /// {
  ///   "ingredients": [...],
  ///   "detections": [...],
  ///   "recommendations": [{"id": "...","metadata": {...}, "distance": ...}, ...]
  /// }
  Future<Map<String, dynamic>> uploadAndRecommend(
    File imageFile, {
    int topK = 5,
  }) async {
    try {
      final uri = Uri.parse("$fastApiPredictAndRecommendUrl?top_k=$topK");
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('image', imageFile.path));

      final resp = await request.send();

      if (resp.statusCode != 200) {
        final errBody = await resp.stream.bytesToString();
        throw HttpException(
          "FastAPI /predict-and-recommend error: ${resp.statusCode} $errBody",
        );
      }

      final respStr = await resp.stream.bytesToString();
      final data = jsonDecode(respStr);
      return (data is Map<String, dynamic>)
          ? data
          : Map<String, dynamic>.from(data as Map);
    } catch (e) {
      print("⚠️ FastAPI 연결 에러 (/predict-and-recommend): $e");
      return {};
    }
  }
}
