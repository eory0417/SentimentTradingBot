# SentimentTradingBot

Freqtrade + FreqAI 기반 하이브리드 트레이딩 봇입니다. 뉴스 감성(RoBERTa), 거래량 화제성, 기술적 지표, FreqAI ML 게이트를 결합합니다.

## 구조

```
user_data/
├── config.json                  # Freqtrade 전체 설정
├── modules/news_provider.py     # RSS 뉴스 수집 (교체 가능 인터페이스)
├── strategies/HybridSentimentStrategy.py
├── execution_engine.py          # 참고용 ccxt 엔진 (미연동)
└── secrets.json                 # API 키 (gitignore)
```

## 설치

```bash
pip install -r requirements.txt
```

`secrets.json`을 `secrets.example.json`을 참고해 생성한 뒤, `config.json`의 `exchange.key` / `exchange.secret` 및 `telegram` 항목에 값을 입력하세요.

## 데이터 다운로드

```bash
freqtrade download-data \
  --config user_data/config.json \
  --days 60 \
  -t 5m 15m 1h
```

## FreqAI 백테스트

```bash
freqtrade backtesting \
  --config user_data/config.json \
  --strategy HybridSentimentStrategy \
  --freqaimodel PyTorchMLPRegressor \
  --timerange 20250101-
```

## 드라이런

```bash
freqtrade trade --config user_data/config.json --dry-run
```

## 진입 로직

1. **뉴스 감성**: `%-sentiment > 0.5` AND `%-hype_intensity > 2.0` (거래량 z-score)
2. **RSI 상승 다이버전스**: 가격 하락 + RSI 상승 + RSI < 35
3. **눌림목 재진입**: EMA20 2% 이내 + RSI > 50 + 거래량 증가

위 3조건 중 하나 + FreqAI `do_predict` / `DI_values` 게이트 통과 시 롱 진입.

## 청산

- RSI > 70 또는 감성 점수 < 0.3
- `stoploss`: -5%
- `minimal_roi`: 단계적 익절

## 뉴스 소스 확장

`user_data/modules/news_provider.py`의 `NewsProvider` 인터페이스를 구현해 `CryptoPanicProvider` 등으로 교체할 수 있습니다.



실행 방법
# 1. 의존성 설치
pip install -r requirements.txt
# 2. secrets.json 키를 config.json exchange/telegram에 입력
# 3. 데이터 다운로드
.\.venv\Scripts\python.exe -m freqtrade download-data --config user_data/config.json --days 60 -t 5m 15m 1h
# 4. FreqAI 백테스트
.\.venv\Scripts\python.exe -m freqtrade backtesting --config user_data/config.json --strategy HybridSentimentStrategy --freqaimodel PyTorchMLPRegressor --timerange 20260401-20260515                                
# 5. 드라이런
.\.venv\Scripts\python.exe run_freqtrade_with_secrets.py --config user_data/config.json --secrets user_data/secrets.json --disable-telegram trade --dry-run --freqaimodel PyTorchMLPRegressor
