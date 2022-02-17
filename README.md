# Waveform & HRV Extraction
-----

# Work Flow

1. Image 필요없는 부분 제거 (그림판에서 흰색으로 지우기)
2. 환자단위 Image 파일 read 
  - Fetal Heart Rate(fhr) : rgb (3 channel)
  - Uterine contraction : gray (1 channel, 반전필요)
4. Image의 대략적인 좌표 설정 (target box의 x축, y축 좌표구해서 추가)
5. phl 알고리즘에 의한 정밀한 box boundary 설정 (Threshold 설정 필요: 너무 높으면 남는 선이 거의 없고, 너무 낮으면 대부분의 선이 선택됨.)
6. Image 전체길이 구함
7. fhr (box 내 픽셀의 std와 green 조건으로 빨간색 line 찾기), uc (box 내 픽셀에서 일정 수치 미만인 검정색 line 찾기)
8. scaler 이용해서 보고싶은 곳만 잘라서 환자ID와 Image pair-wise 하게 저장
9. 위 과정에서 구한 waveform을 csv 로 저장
10. fhr 에서 nni 구한 후, HRV 계산하여 csv 로 저장
