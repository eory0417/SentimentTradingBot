import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, merge_informative_pair
from freqtrade.persistence import Trade

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.news_provider import pair_to_base_symbol

logger = logging.getLogger(__name__)

SENTIMENT_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# 🔧 개선: 감성 점수 매핑 강화
LABEL_SCORES = {
    "positive": 1.0,
    "neutral": 0.5,
    "negative": 0.0
}


class HybridSentimentStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1m"                             
    can_short = True                              
    startup_candle_count = 1000                  
    process_only_new_candles = True

    # config.json 기반의 정밀 제어를 위해 네이티브 고정 기능들은 최대 영역으로 우회시킵니다.
    use_custom_stoploss = True  
    stoploss = -0.99                             
    trailing_stop = False       # custom_exit 함수에서 롱/숏 분리형 트레일링 스톱을 직접 작동시킵니다.

    # [하이퍼옵트 탐색 공간 조율]
    ai_long_conviction = DecimalParameter(0.0, 0.015, default=0.007, space="buy", optimize=True)
    hype_threshold = DecimalParameter(1.0, 3.0, default=1.65, space="buy", optimize=True) # 💡 [수정] Z-Score 기준을 1.092 ➔ 1.65로 격상 (평균 이상의 강력한 화력 필요)
    rsi_long_floor = IntParameter(25, 45, default=25, space="buy", optimize=True)
    rsi_short_roof = IntParameter(62, 80, default=75, space="buy", optimize=True)
    
    # 중립 뉴스 필터 최적화 하한선 바운더리 격상
    sentiment_threshold = DecimalParameter(0.55, 0.85, default=0.68, space="buy", optimize=True) # 💡 [수정] 롱 진입 장벽을 0.55 ➔ 0.68로 격상 (확실한 호재만 인정)

    DI_threshold = 1.0
    sentiment_exit_threshold = 0.3
    rsi_exit_threshold = 70

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._sentiment_pipeline = None
        # 🔧 변경: RSSNewsProvider → HybridNewsProvider (RSS + CoinGecko)
        from modules.news_provider import HybridNewsProvider
        self._news_provider = HybridNewsProvider()
        # config.json의 커스텀 파라미터 셋 로드
        self.custom_params = config.get("custom_trader_params", {})

    @property
    def sentiment_pipeline(self):
        if self._sentiment_pipeline is None:
            from transformers import pipeline
            self._sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, truncation=True, max_length=512)
        return self._sentiment_pipeline

    def leverage(self, pair: str, current_time, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        # config.json의 leverage_value 설정을 실시간 연동
        return min(float(self.custom_params.get("leverage_value", 1)), max_leverage)

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, "5m") for pair in pairs]

# ==============================================================================
    # 🛡️ 리팩토링된 안전 손절망 (ATR 소음 필터링 및 고정 손절 매칭)
    # ==============================================================================
    def custom_stoploss(self, pair: str, trade, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs) -> float:
        # 고정 퍼센트 손절: 설정된 stoploss 값을 그대로 사용합니다.
        return float(self.custom_params.get("short_stoploss", -0.02) if trade.is_short else self.custom_params.get("long_stoploss", -0.02))


    # ==============================================================================
    # 🎯 [수정 완료] LocalTrade 호환성 확보형 익절 및 트레일링 스톱 제어
    # ==============================================================================
    def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> str | bool | None:
        # 🎯 LocalTrade 및 실전 Trade 객체 모두에서 최고 수익률을 안전하게 연산해 내는 표준 인터페이스 기법
        if trade.is_short:
            highest_profit = trade.calc_profit_ratio(trade.min_rate)
            take_profit = float(self.custom_params.get("short_take_profit", 0.04))
            trailing_retrace = float(self.custom_params.get("short_trailing_drop", 0.5))
            trailing_retrace = min(max(trailing_retrace, 0.0), 1.0)

            if highest_profit <= -take_profit:
                # 최고 수익 대비 리트레이스 비율로 청산
                threshold_profit = highest_profit * (1.0 - trailing_retrace)
                if current_profit >= threshold_profit:
                    return "short_trailing_stop"
        else:
            highest_profit = trade.calc_profit_ratio(trade.max_rate)
            take_profit = float(self.custom_params.get("long_take_profit", 0.04))
            trailing_retrace = float(self.custom_params.get("long_trailing_drop", 0.5))
            trailing_retrace = min(max(trailing_retrace, 0.0), 1.0)

            if highest_profit >= take_profit:
                # 최고 수익 대비 리트레이스 비율로 청산
                threshold_profit = highest_profit * (1.0 - trailing_retrace)
                if current_profit <= threshold_profit:
                    return "long_trailing_stop"

        # ⏱️ 🔧 타임아웃 4시간으로 연장 (기존 2시간 → 14400초)
        trade_duration_seconds = (current_time - trade.open_date_utc).total_seconds()
        if trade_duration_seconds >= 14400:
            return "timeout_4h"
            
        return None

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, metadata: dict, **kwargs) -> pd.DataFrame:
        dataframe[f"%-rsi-period_{period}"] = ta.RSI(dataframe, timeperiod=period)
        dataframe[f"%-ema-period_{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%-volume_mean-period_{period}"] = dataframe["volume"].rolling(period).mean()
        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, metadata: dict, **kwargs) -> pd.DataFrame:
        if "sentiment" not in dataframe.columns:
            dataframe["sentiment"] = self.get_roberta_score(dataframe, metadata)
        if "hype_intensity" not in dataframe.columns:
            dataframe["hype_intensity"] = self.calculate_hype(dataframe)

        dataframe["%-sentiment"] = dataframe["sentiment"]
        dataframe["%-hype_intensity"] = dataframe["hype_intensity"]
        dataframe["%-crypto_close"] = dataframe["close"]

        label_period = 120
        dataframe["&-s_close"] = dataframe["close"].shift(-label_period) / dataframe["close"] - 1
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # 🔧 새로 추가: MACD 지표 (진입 신뢰도 향상)
        macd_result = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd_result['macd']
        dataframe['macd_signal'] = macd_result['macdsignal']
        dataframe['macd_diff'] = macd_result['macdhist']
        
        # 1. 거래소로부터 깨끗한 5분봉(5m) 데이터를 별도로 빌려옵니다.
        informative = self.dp.get_pair_dataframe(pair, "5m")
        
        # 2. [5분봉 매크로 기준] 기술적 지표 연산 수행
        informative['rsi_5m'] = ta.RSI(informative, timeperiod=14)
        informative['ema20_5m'] = ta.EMA(informative, timeperiod=20)
        informative['ema1000_5m'] = ta.EMA(informative, timeperiod=1000)
        informative['atr_5m'] = ta.ATR(informative, timeperiod=14)
        
        # [매크로 거래량 수선] 1분봉 소음을 필터링할 5분봉 기준 볼륨 Z-Score 스코어링 선행 연산
        rolling_mean_5m = informative["volume"].rolling(20).mean()
        rolling_std_5m = informative["volume"].rolling(20).std().replace(0, np.nan)
        informative['hype_5m'] = ((informative["volume"] - rolling_mean_5m) / rolling_std_5m).fillna(0)
        
        # 3. Freqtrade 내장 엔진을 통해 5분봉 데이터셋을 1분봉 본진 차트에 병합
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, "5m", ffill=True)
        
        # 4. 1분봉 데이터프레임 구조에 실시간 파이프라인 정렬 동기화
        dataframe["sentiment"] = self.get_roberta_score(dataframe, metadata)
        # 1분봉의 소음성 volume 연산값 대신, 5분봉 매크로 격상 데이터인 hype_5m_5m 으로 오버라이딩 체인지합니다.
        dataframe["hype_intensity"] = dataframe["hype_5m_5m"]

        dataframe = dataframe.fillna(0)
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair", "Unknown")
        # Avoid generating new entry signals for pairs that already have an open trade
        try:
            open_for_pair = any(t.pair == pair for t in Trade.get_open_trades())
        except Exception:
            open_for_pair = False
        if hasattr(self, 'config') and self.config.get('runmode') in ['live', 'dry_run']:
            from datetime import datetime, timedelta, timezone
            now = datetime.now(timezone.utc)
            just_now_lines = self._news_provider.fetch_headlines(pair, now - timedelta(minutes=2), now)
            
            if just_now_lines:
                logger.info(f"==========================================================================")
                logger.info(f"📡 [1분 주기 실시간 속보 감지] {pair} 관련 {len(just_now_lines)}개의 뉴스를 포착했습니다.")
                for idx, h in enumerate(just_now_lines, 1):
                    logger.info(f"   👉 속보 제목 [{idx}]: {h.title}")
                
                texts = [h.title for h in just_now_lines]
                results = self.sentiment_pipeline(texts)
                scores = []
                logger.info(f"--------------------------------------------------------------------------")
                for text, res in zip(texts, results):
                    label = res.get("label", "neutral").lower()
                    confidence = res.get("score", 0.0)
                    mapped_score = LABEL_SCORES.get(label, 0.5)
                    scores.append(mapped_score)
                    logger.info(f"   🧠 [AI 연산 완료] 요약: '{text[:35]}...'\n      ➔ 모델 분류: {label.upper()} (확신도: {confidence:.4f}) | 매칭 점수: {mapped_score}")
                
                realtime_sentiment = sum(scores) / len(scores)
                dataframe.loc[dataframe.index[-1], 'sentiment'] = realtime_sentiment
                logger.info(f"   📊 [데이터 동기화] 1분봉 실시간 감성 총점 주입 완료: {realtime_sentiment:.3f}")
                logger.info(f"==========================================================================")

        # 🔧 버그 수정: do_predict, DI_values 컬럼 안전성 확인
        if "do_predict" not in dataframe.columns:
            dataframe["do_predict"] = 1  # freqai 비활성화 시 기본값 = 1
        if "DI_values" not in dataframe.columns:
            dataframe["DI_values"] = 0.5  # freqai 비활성화 시 기본값 = 0.5
        
        # 🔧 sentiment 기본값 상향 (뉴스 부족 시에도 거래 가능하게)
        dataframe["sentiment"] = dataframe["sentiment"].fillna(0.60)  # 0.5 → 0.60
        
        # 🔧 hype_intensity 기본값 (Z-Score 계산이 0이 되는 초기 단계 처리)
        dataframe["hype_intensity"] = dataframe["hype_intensity"].fillna(1.0)  # 1.0 이상 유지

        ai_cols = [c for c in dataframe.columns if c.startswith("&do_predict_") and c != "&do_predict"]
        if ai_cols:
            pred_col = ai_cols[0]
            ai_long_ok = dataframe[pred_col] > self.ai_long_conviction.value
            ai_short_ok = dataframe[pred_col] < 0.0
        else:
            ai_long_ok = ai_short_ok = True

        macro_bullish = dataframe["close"] > dataframe["ema1000_5m_5m"].fillna(dataframe["close"])
        macro_bearish = dataframe["close"] < dataframe["ema1000_5m_5m"].fillna(dataframe["close"])

        # --------------------------------------------------------------------------
        # 📈 개선된 LONG 진입 조건식 (OR 로직으로 충분히 유연하게)
        # --------------------------------------------------------------------------
        news_signal = (
            (dataframe["sentiment"] >= self.sentiment_threshold.value)
            & (dataframe["hype_intensity"] > self.hype_threshold.value)
        )
        
        bullish_div = (
            (dataframe["close"] < dataframe["close"].shift(1))
            & (dataframe["rsi_5m_5m"] > dataframe["rsi_5m_5m"].shift(1))
            & (dataframe["rsi_5m_5m"] < self.rsi_long_floor.value)
        )
        
        pullback_entry = (
            (dataframe["close"] <= dataframe["ema20_5m_5m"] * 1.02)
            & (dataframe["rsi_5m_5m"] > 50)
            & (dataframe["volume"] > dataframe["volume"].rolling(20).mean())
        )
        
        # MACD 상승 신호 추가 (진입 신뢰도 향상)
        macd_bullish = (
            (dataframe["macd"] > dataframe["macd_signal"])
            & (dataframe["macd_diff"] > 0)
        )

        latest_index = dataframe.index[-1]
        recent_uptrend = dataframe["close"].shift(20).loc[latest_index]
        recent_uptrend = False if pd.isna(recent_uptrend) else dataframe["close"].loc[latest_index] > recent_uptrend

        latest_conditions = {
            "macro_bullish": bool(macro_bullish.loc[latest_index]),
            "news_signal": bool(news_signal.loc[latest_index]),
            "bullish_div": bool(bullish_div.loc[latest_index]),
            "pullback_entry": bool(pullback_entry.loc[latest_index]),
            "macd_bullish": bool(macd_bullish.loc[latest_index]),
            "recent_uptrend": bool(recent_uptrend),
            "sentiment": float(dataframe["sentiment"].loc[latest_index]),
            "hype_intensity": float(dataframe["hype_intensity"].loc[latest_index]),
            "rsi_5m_5m": float(dataframe["rsi_5m_5m"].loc[latest_index]),
            "close": float(dataframe["close"].loc[latest_index])
        }

        logger.info("===== 포지션 진입 상태 확인 =====")
        logger.info(
            f"    {pair} 진입조건: 매크로상승={latest_conditions['macro_bullish']}, "
            f"뉴스신호={latest_conditions['news_signal']}, 골든다이버전스={latest_conditions['bullish_div']}, "
            f"풀백={latest_conditions['pullback_entry']}, MACD상승={latest_conditions['macd_bullish']}, "
            f"최근상승={latest_conditions['recent_uptrend']}, sentiment={latest_conditions['sentiment']:.2f}, "
            f"hype={latest_conditions['hype_intensity']:.2f}, rsi5m={latest_conditions['rsi_5m_5m']:.1f}, "
            f"close={latest_conditions['close']:.2f}"
        )
        entry_signal = (
            macro_bullish.loc[latest_index]
            and (news_signal.loc[latest_index] or bullish_div.loc[latest_index] or pullback_entry.loc[latest_index] or macd_bullish.loc[latest_index])
            and recent_uptrend
        )
        if entry_signal:
            signal_reasons = []
            if latest_conditions['news_signal']:
                signal_reasons.append('뉴스')
            if latest_conditions['bullish_div']:
                signal_reasons.append('골든다이버전스')
            if latest_conditions['pullback_entry']:
                signal_reasons.append('풀백')
            if latest_conditions['macd_bullish']:
                signal_reasons.append('MACD')
            reason_text = ' + '.join(signal_reasons)
            if open_for_pair:
                logger.info(f"    ⛔ 진입스킵(이미 오픈포지션): {pair} ({reason_text})")
            else:
                logger.info(f"    ✅ 진입허용: {pair} ({reason_text})")
        else:
            failure_reasons = []
            if not latest_conditions["macro_bullish"]:
                failure_reasons.append("매크로 하락")
            if not (latest_conditions["news_signal"] or latest_conditions["bullish_div"] or latest_conditions["pullback_entry"] or latest_conditions["macd_bullish"]):
                failure_reasons.append("진입 신호 부족")
            if not latest_conditions["recent_uptrend"]:
                failure_reasons.append("최근 상승 부재")
            logger.info(f"    ❌ 진입불가: {pair} - 사유: {', '.join(failure_reasons)}")

        # 🔧 개선: AND 체인 단순화 - do_predict 제거, 신호 조건만 유지
        if not open_for_pair:
            dataframe.loc[
                macro_bullish
                & (news_signal | bullish_div | pullback_entry | macd_bullish)
                & (dataframe["close"] > dataframe["close"].shift(20)),  # 최근 20분 상승 추세
                ["enter_long", "enter_tag"]
            ] = (1, "Long_Hybrid")

            if entry_signal:
                logger.info(f"    ▶ {pair} 장기 진입 신호 생성됨 (Long_Hybrid)")
        else:
            logger.debug(f"Skipped entry assignment for {pair} because an open trade exists")

        # --------------------------------------------------------------------------
        # 📉 개선된 SHORT 진입 조건식
        # --------------------------------------------------------------------------
        bearish_news_signal = (
            (dataframe["sentiment"] <= (1.0 - self.sentiment_threshold.value))
            & (dataframe["hype_intensity"] > self.hype_threshold.value)
        )
        
        bearish_div = (
            (dataframe["close"] > dataframe["close"].shift(1))
            & (dataframe["rsi_5m_5m"] < dataframe["rsi_5m_5m"].shift(1))
            & (dataframe["rsi_5m_5m"] > self.rsi_short_roof.value)
        )
        
        trend_short = (
            (dataframe["close"] <= dataframe["ema20_5m_5m"])
            & (dataframe["rsi_5m_5m"] < 45)
            & (dataframe["volume"] > dataframe["volume"].rolling(20).mean())
        )
        
        # MACD 하락 신호 추가
        macd_bearish = (
            (dataframe["macd"] < dataframe["macd_signal"])
            & (dataframe["macd_diff"] < 0)
        )

        if not open_for_pair:
            dataframe.loc[
                macro_bearish
                & (bearish_news_signal | bearish_div | trend_short | macd_bearish)
                & (dataframe["close"] < dataframe["close"].shift(20)),  # 최근 20분 하락 추세
                ["enter_short", "enter_tag"]
            ] = (1, "Short_Hybrid")
        else:
            logger.debug(f"Skipped short entry assignment for {pair} because an open trade exists")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        sentiment_reversal = dataframe["sentiment"] < self.sentiment_exit_threshold
        rsi_overbought = dataframe["rsi_5m_5m"] > self.rsi_exit_threshold
        dataframe.loc[sentiment_reversal | rsi_overbought, "exit_long"] = 1

        short_sentiment_reversal = dataframe["sentiment"] > 0.65
        rsi_oversold = dataframe["rsi_5m_5m"] < 30
        dataframe.loc[short_sentiment_reversal | rsi_oversold, "exit_short"] = 1
        return dataframe

    def get_roberta_score(self, df: pd.DataFrame, metadata: dict) -> pd.Series:
        pair = metadata.get("pair", "")
        if df.empty or "date" not in df.columns:
            return pd.Series(0.5, index=df.index)
        since = pd.Timestamp(df["date"].iloc[0]).to_pydatetime()
        until = pd.Timestamp(df["date"].iloc[-1]).to_pydatetime()
        try:
            headlines = self._news_provider.fetch_headlines(pair, since, until)
        except Exception as e:
            return pd.Series(0.5, index=df.index)
        if not headlines:
            return pd.Series(0.5, index=df.index)
        scores_by_time = []
        texts = [h.title for h in headlines]
        try:
            results = self.sentiment_pipeline(texts)
            for item, result in zip(headlines, results):
                label = result.get("label", "neutral").lower()
                score = LABEL_SCORES.get(label, 0.5)
                scores_by_time.append((pd.Timestamp(item.published), score))
        except Exception as e:
            return pd.Series(0.5, index=df.index)
        if not scores_by_time:
            return pd.Series(0.5, index=df.index)
        scores_df = pd.DataFrame(scores_by_time, columns=["date", "score"]).sort_values("date").drop_duplicates("date", keep="last")
        scores_df["date"] = pd.to_datetime(scores_df["date"]).astype("datetime64[us, UTC]")
        candle_dates = pd.to_datetime(df["date"]).astype("datetime64[us, UTC]")
        merged = pd.merge_asof(pd.DataFrame({"date": candle_dates, "_idx": df.index}), scores_df, on="date", direction="backward")
        return merged["score"].fillna(0.5).set_axis(df.index)

    def calculate_hype(self, df: pd.DataFrame) -> pd.Series:
        rolling_mean = df["volume"].rolling(20).mean()
        rolling_std = df["volume"].rolling(20).std().replace(0, np.nan)
        return ((df["volume"] - rolling_mean) / rolling_std).fillna(0)