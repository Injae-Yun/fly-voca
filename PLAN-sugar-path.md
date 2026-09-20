# 계획: BANC에서 설탕 신호가 다리에 닿지 않는 버그

작성: 2026-09-20 · 담당: 실행 에이전트 · 검수: 상위 세션

## 증상

BANC 전 CNS 그래프에서 looming은 다리 운동뉴런 38개를 움직이는데
(`accessory_tibia_flexor` d=+17 등), **설탕(LB3*)과 CO2(ORN_V)는 0~1개**다.
FAFB에서는 설탕 → MN9(주둥이)가 검증된 벤치마크였으므로(문서 01·02),
BANC 전환에서 무언가가 깨졌다.

## 확정된 사실 (재검증 불필요)

- 반증됨: w_syn 창(0.05~0.085에 없음) · 히스타민 부호(시엽은 상대적으로 식음)
  · 약연결 컷(KC 그물 68→24%에도 KC 발화 71→63%) · KC 역치(+6mV에도
  다리MN 기저 2.33 Hz 불변)
- **KC 포화와 다리MN 기저는 분리된 현상**이다. "뇌 포화로 신호가 묻힘"은 틀렸다.
- 미검증: **가설① — BANC은 엣지별 전달물질이 없어 뉴런 단위(데일)로 부호를
  칠했다. FAFB는 엣지별이었다.**
- looming은 통과한다 = 배선·시뮬레이터 자체는 산 것. 양성 대조로 쓸 것.

## 환경

- 본 환경 파이썬(3.11)에 `flyvoca` editable 설치됨. GPU torch cu130.
- 작업 루트 `D:\Code\fly_voca`. 스크립트는 `bodysnatch/scripts/`.
- BANC 그래프: `from flyvoca.banc_graph import build, select, leg_motor,
  spontaneous_groups` — build()는 ~7초, 시뮬 600ms×12trial ~25초.
- FAFB 그래프: `from flyvoca.graph import build` (엣지별 부호).
- 시뮬레이터: `from bodysnatch.sim import Brain, Params`.
  BANC은 `Brain(W, Params(w_syn=0.16, sigma_v=2.0, r_spont=0.0))` +
  `brain.spont_groups = spontaneous_groups(meta)`.
- `.venv-body`는 flygym 전용 — 이 작업에는 불필요.

### 실행 수칙 (전부 실제로 겪은 함정)

- 장기 실행은 `python -u script.py > 로그 2>&1`로 파일에 직접. grep을 파이프에
  끼우면 버퍼링으로 로그가 빈다.
- heredoc으로 파이썬 파일 쓰지 말 것(따옴표 사고 잦음). Write 도구 사용.
- 조건 간 **시드를 동일하게** — 짝지은 비교여야 한다.
- 효과 크기는 Cohen's d (시행 분산 기반), sham은 다른 시드의 기저.
- 숫자는 있는 그대로 보고. 반증도 결과다. 통과시키려고 파라미터를 조정하지 말 것.

## 작업

### T1. 트레이스 스크립트 수리 및 실행 (최우선)

`bodysnatch/scripts/86_trace_sugar.py` 74행 f-string 문법 오류:

```python
# 현재 (깨짐):
print(f"  hop {h}: reaches {int((step[dn] > 0).sum():,} of {len(dn)} DNs, "
# 수정:
n_hit = int((step[dn] > 0).sum())
print(f"  hop {h}: reaches {n_hit:,} of {len(dn)} DNs, "
```

실행하고 산출물 확보:
- 단계별 표 (stimulated → SEZ/brain → descending → VNC → leg motor) ×
  (sugar, looming, co2): base_Hz / stim_Hz / max|d| / n_d>2
- 순수 배선 홉 체크: LB3 → DN 도달 가중치 (hop 1~4)

**판정 기준**: 신호가 죽는 첫 단계를 특정한다.
- 배선 홉에서 이미 0 → T2로
- 배선은 닿는데 동역학이 SEZ→DN에서 죽음 → T3로
- DN은 움직이는데 VNC/다리에서 죽음 → T4로

### T2. (배선이 원인일 때) 미주석 뉴런 경유 확인

BANC엔 `super_class=None`이 28,632개다. LB3 → DN 경로가 이들을 경유하는지:
- LB3에서 2홉 이내 도달 뉴런의 super_class 분포
- None 뉴런의 nt_sign 분포 (미상→흥분성 기본값이 여기서 어떻게 작용하는지)

### T3. (동역학이 원인일 때) 구동 강도 스윕

문서 09b에서 식초는 400Hz에서야 명확해졌다. 같은 방식:
- sugar를 r_poi = 100/200/400 Hz로, 각 12시행
- 각 단계(SEZ/DN/VNC/legMN)의 n_d>2가 용량을 따라 자라는지
- 자라면 "약하지만 실재" — 문서 09b와 같은 결론. 안 자라면 T2 재검토.

### T4. 가설① 판정 — 데일 근사가 검증된 벤치마크를 깨는가 (결정적)

FAFB에서는 엣지별 부호로 설탕→MN9 = 71.9 Hz가 레퍼런스(67.0)와 맞았다.
**같은 FAFB 데이터를 뉴런 단위 부호로 다시 칠하고** 같은 실험을 돌린다:

- `flyvoca/graph.py`의 build를 복사·수정해 `nt_type`(neurons.csv의 뉴런 단위
  예측)으로 모든 출력 엣지에 부호를 칠하는 변형 `build_dale()`을
  **별도 스크립트 안에** 만들 것 (graph.py 자체는 수정 금지)
- ref-20 설탕 뉴런(문서 02의 REF_SUGAR, bodysnatch/reference.py에 있음),
  w_syn=0.275, sigma_v=0, r_spont=0, 100Hz, 30시행 → MN9 발화율
- **판정**: 엣지별 71.9 Hz 대비 뉴런 단위가 크게 어긋나면(예: >±30%)
  가설①이 BANC 문제의 유력 원인. 비슷하면 가설①도 기각이고
  원인은 BANC 고유(계수 규약·주석 결손)로 좁혀진다.

### T5. 회귀 확인

무엇을 고치든:
- looming → 다리 38±α개 유지되는가
- (T4에서 그래프를 건드렸으면) FAFB 설탕→MN9 벤치마크 유지되는가

## 산출물

- 로그: 스크래치패드가 아니라 `D:\Code\fly_voca\data\cache\debug\` 에 남길 것
- 보고서: `bodysnatch/REPORT-sugar-path.md` — 각 T의 실측 숫자, 판정,
  그리고 **어떤 가설이 어떤 근거로 살았/죽었는지**. 추측과 측정을 구분해 쓸 것.
- 커밋은 하지 말 것 (검수 후 상위 세션이 수행)
