## [kugelaudio-python-sdk-v2.1.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v2.0.1...python-sdk-v2.1.0) (2026-09-23)

## [kugelaudio-python-sdk-v2.0.1](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v2.0.0...python-sdk-v2.0.1) (2026-09-22)

## [kugelaudio-python-sdk-v2.0.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.11.0...python-sdk-v2.0.0) (2026-09-21)

## [kugelaudio-python-sdk-v1.11.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.10.0...python-sdk-v1.11.0) (2026-09-07)

## [kugelaudio-python-sdk-v1.10.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.9.0...python-sdk-v1.10.0) (2026-09-05)

## [kugelaudio-python-sdk-v1.9.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.8.0...python-sdk-v1.9.0) (2026-08-07)

### Features

* **js-sdk,python-sdk:** add speed and bound prewarm timeout in the LiveKit plugins ([#1823](https://github.com/Kugelaudio/KugelAudio/issues/1823)) ([94bd656](https://github.com/Kugelaudio/KugelAudio/commit/94bd6564a867c7910d73ab12b422f6129f420636))

## [kugelaudio-python-sdk-v1.8.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.7.0...python-sdk-v1.8.0) (2026-07-27)

### Features

* **turn-detection,ingress:** protected KugelTurn customer runtime + licensing ([#1642](https://github.com/Kugelaudio/KugelAudio/issues/1642)) ([ac51591](https://github.com/Kugelaudio/KugelAudio/commit/ac51591f9ae09817abd3a519127241d15b054c74))

## [kugelaudio-python-sdk-v1.7.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.6.0...python-sdk-v1.7.0) (2026-07-21)

### Features

* **ingress:** per-span <prosody rate> speed control ([#1373](https://github.com/Kugelaudio/KugelAudio/issues/1373)) ([7861e88](https://github.com/Kugelaudio/KugelAudio/commit/7861e881cc9d89730f01681f72192ad7a6f3d2b8))
* train and validate KugelTurn v0.6 ([#1576](https://github.com/Kugelaudio/KugelAudio/issues/1576)) ([166dc4b](https://github.com/Kugelaudio/KugelAudio/commit/166dc4b0ce8d347c076e4cd049e07b3f2f2b7818))
* update session settings per turn (KUG-1166) ([#1500](https://github.com/Kugelaudio/KugelAudio/issues/1500)) ([7521d35](https://github.com/Kugelaudio/KugelAudio/commit/7521d35d56925f2f76d6ac19f40bdd735680584c))

### Bug Fixes

* **tts:** word timestamps — AlignAudio crashed on every call; docs claimed REST support ([#1374](https://github.com/Kugelaudio/KugelAudio/issues/1374)) ([2fbd016](https://github.com/Kugelaudio/KugelAudio/commit/2fbd016cbf2de2d5d6705c227bd8f75c5635259d))

### Code Refactoring

* **ingress:** reorganize routes and metadata caching ([#1342](https://github.com/Kugelaudio/KugelAudio/issues/1342)) ([91ce468](https://github.com/Kugelaudio/KugelAudio/commit/91ce46897a1a81e01a05d68f3b9fc00af90b885b))

## [kugelaudio-python-sdk-v1.6.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.5.0...python-sdk-v1.6.0) (2026-06-10)

### Features

* **ingress,python-sdk,js-sdk,java-sdk:** per-session usage over WebSocket ([#1346](https://github.com/Kugelaudio/KugelAudio/issues/1346)) ([2881881](https://github.com/Kugelaudio/KugelAudio/commit/28818816dca9c8d222391691d70f458c0eb28ed8))
* **ingress,python-sdk,js-sdk:** streaming final end-of-audio frame ([#1362](https://github.com/Kugelaudio/KugelAudio/issues/1362)) ([3fa95d2](https://github.com/Kugelaudio/KugelAudio/commit/3fa95d2f8597e6c9ced0aaf8370682dbcb123c71))
* **ingress:** output_format token + server-side G.711 (KUG-1190) ([#1345](https://github.com/Kugelaudio/KugelAudio/issues/1345)) ([3723291](https://github.com/Kugelaudio/KugelAudio/commit/372329196c4c91aa41fe2111783872874b6e895b))
* per-request dictionary selection (KUG-1094) ([#1361](https://github.com/Kugelaudio/KugelAudio/issues/1361)) ([3c28968](https://github.com/Kugelaudio/KugelAudio/commit/3c28968d32018bf3cafe1d312f32831668ea96b8))

## [kugelaudio-python-sdk-v1.5.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.4.4...python-sdk-v1.5.0) (2026-06-06)

### Features

* **ingress:** add request observability metadata ([#1321](https://github.com/Kugelaudio/KugelAudio/issues/1321)) ([a9c5178](https://github.com/Kugelaudio/KugelAudio/commit/a9c5178193cb8b746a8bbd9b566b11f7b1d00f6d))
* **sdks:** default all SDKs to kugel-3 model ([#1323](https://github.com/Kugelaudio/KugelAudio/issues/1323)) ([c4de212](https://github.com/Kugelaudio/KugelAudio/commit/c4de212c91e16326a15dbee5622acacc83ed85bb))

## [kugelaudio-python-sdk-v1.4.4](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.4.3...python-sdk-v1.4.4) (2026-06-04)

### Bug Fixes

* **python-sdk:** propagate ingress errors through SDK integrations ([#1313](https://github.com/Kugelaudio/KugelAudio/issues/1313)) ([3ae2e03](https://github.com/Kugelaudio/KugelAudio/commit/3ae2e03745b49cca0712c20d9a658c160f4b6f38))

## [kugelaudio-python-sdk-v1.4.3](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.4.2...python-sdk-v1.4.3) (2026-06-01)

### Bug Fixes

* **python-sdk:** pipecat 0.x baseline + per-turn context isolation (KUG-1123) ([#1277](https://github.com/Kugelaudio/KugelAudio/issues/1277)) ([bbdc55b](https://github.com/Kugelaudio/KugelAudio/commit/bbdc55bb44374299ebeebe4cff01cd4d8524fe2b))

## [kugelaudio-python-sdk-v1.4.2](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.4.1...python-sdk-v1.4.2) (2026-05-30)

### Performance Improvements

* **python-sdk:** pre-provision Pipecat turn context to hide WS TTFA ([#1237](https://github.com/Kugelaudio/KugelAudio/issues/1237)) ([73b0950](https://github.com/Kugelaudio/KugelAudio/commit/73b0950238a8baa6cc85b5e477f50afa265ef240))

## [kugelaudio-python-sdk-v1.4.1](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.4.0...python-sdk-v1.4.1) (2026-05-30)

### Bug Fixes

* harden /ws/tts/multi context cap (server) + close Pipecat 1.x per-turn contexts (SDK) ([#1230](https://github.com/Kugelaudio/KugelAudio/issues/1230)) ([6f5fd1a](https://github.com/Kugelaudio/KugelAudio/commit/6f5fd1aa3e0c9e1f2cc1310de4ca1a56acb3fd34))

## [kugelaudio-python-sdk-v1.4.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.3.2...python-sdk-v1.4.0) (2026-05-29)

### Features

* streaming barge-in (cancelCurrent) across server + JS/Python/Java SDKs ([#1210](https://github.com/Kugelaudio/KugelAudio/issues/1210)) ([341e54f](https://github.com/Kugelaudio/KugelAudio/commit/341e54f169b4dd9242272b249fca30f005bfc3b8))

### Bug Fixes

* **ingress:** bound whole-text flush chunk size to prevent zero audio (KUG-1082) ([#1226](https://github.com/Kugelaudio/KugelAudio/issues/1226)) ([7cb30d7](https://github.com/Kugelaudio/KugelAudio/commit/7cb30d750cd7a5c4a600e9fe2efcee9bedb7d734))

## [kugelaudio-python-sdk-v1.3.2](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.3.1...python-sdk-v1.3.2) (2026-05-28)

### Bug Fixes

* **python-sdk:** stop Pipecat multi-context leak that drops calls at ~30-40s ([#1205](https://github.com/Kugelaudio/KugelAudio/issues/1205)) ([a006525](https://github.com/Kugelaudio/KugelAudio/commit/a0065259dc92358d71b37767a7b0891580ee172d))

## [kugelaudio-python-sdk-v1.3.1](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.3.0...python-sdk-v1.3.1) (2026-05-21)

### Bug Fixes

* **python-sdk:** default LiveKit word_timestamps off for kugel-2.5 ([#924](https://github.com/Kugelaudio/KugelAudio/issues/924)) ([c061a94](https://github.com/Kugelaudio/KugelAudio/commit/c061a94d9cde2e59733c72b860bbeeca63829c5c))
* **python-sdk:** Pipecat set_voice invalidates persistent session (KUG-760) ([#927](https://github.com/Kugelaudio/KugelAudio/issues/927)) ([38648ec](https://github.com/Kugelaudio/KugelAudio/commit/38648ec0265be129b305af38fc3003143cbb2216))

## [kugelaudio-python-sdk-v1.3.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.2.3...python-sdk-v1.3.0) (2026-05-21)

### Features

* **ingress,sdks:** public API for custom word dictionaries (KUG-765) ([#875](https://github.com/Kugelaudio/KugelAudio/issues/875)) ([9988924](https://github.com/Kugelaudio/KugelAudio/commit/99889244997d1cb4dba9714e2633d84ace9852a3))
* **sdk:** add NotFoundError for unknown resources (KUG-423) ([#872](https://github.com/Kugelaudio/KugelAudio/issues/872)) ([d613b0f](https://github.com/Kugelaudio/KugelAudio/commit/d613b0fa1314c9e9e1f2af924652d94014626ddc))

### Bug Fixes

* **python-sdk:** add drain() to recover tail audio at session close (KUG-421) ([#883](https://github.com/Kugelaudio/KugelAudio/issues/883)) ([d493613](https://github.com/Kugelaudio/KugelAudio/commit/d4936130761af1d3c46aa0829fb2c73fd471374a))

### Reverts

* undo accidental merge of PR [#723](https://github.com/Kugelaudio/KugelAudio/issues/723) ([e8eff2e](https://github.com/Kugelaudio/KugelAudio/commit/e8eff2e86c76b893262782ff6ae763ed405396ce))

## [kugelaudio-python-sdk-v1.2.3](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.2.2...python-sdk-v1.2.3) (2026-05-14)

### Bug Fixes

* **python-sdk,js-sdk,java-sdk:** update regional endpoints ([#660](https://github.com/Kugelaudio/KugelAudio/issues/660)) ([b9a32c0](https://github.com/Kugelaudio/KugelAudio/commit/b9a32c09813c3e9de34a9d0a84ed0e024e1fe158))
* **python-sdk:** recover from dead WebSocket between Pipecat turns ([#695](https://github.com/Kugelaudio/KugelAudio/issues/695)) ([ec7c731](https://github.com/Kugelaudio/KugelAudio/commit/ec7c7313bbc4d9ecb2855f2bffd4ac3b8b9a6a44))

## [kugelaudio-python-sdk-v1.2.2](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.2.1...python-sdk-v1.2.2) (2026-05-10)

### Bug Fixes

* **sdks,ci:** unblock Deploy:SDKs pipeline (stale repo URLs + dry_run default) ([#637](https://github.com/Kugelaudio/KugelAudio/issues/637)) ([a6d66bb](https://github.com/Kugelaudio/KugelAudio/commit/a6d66bb6117d7f48434dbb1324730c0c290c63e5)), closes [#273](https://github.com/Kugelaudio/KugelAudio/issues/273) [#273](https://github.com/Kugelaudio/KugelAudio/issues/273)

## [kugelaudio-python-sdk-v1.2.1](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.2.0...python-sdk-v1.2.1) (2026-05-10)

### Bug Fixes

* **python-sdk:** route Pipecat TTS through multi endpoint ([#564](https://github.com/Kugelaudio/KugelAudio/issues/564)) ([2b9e429](https://github.com/Kugelaudio/KugelAudio/commit/2b9e42998a1177a12daf0f7ced3e9dda19f24b10))

## [kugelaudio-python-sdk-v1.2.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.1.2...python-sdk-v1.2.0) (2026-04-28)

### Features

* **sdks:** unify error classification with actionable messages ([#315](https://github.com/Kugelaudio/KugelAudio/issues/315)) ([ffdcc05](https://github.com/Kugelaudio/KugelAudio/commit/ffdcc054c9e3899230db5148e74228e3c7d94f79))
* **tts:** German digit-chain splitter + CFG aux-slot leak fix ([#324](https://github.com/Kugelaudio/KugelAudio/issues/324)) ([46d9943](https://github.com/Kugelaudio/KugelAudio/commit/46d9943f6a2a090a7db684786d8b43c988f41fba))

## [kugelaudio-python-sdk-v1.1.2](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.1.1...python-sdk-v1.1.2) (2026-04-19)

### Bug Fixes

* **tts,python-sdk:** bounded barge-in cancel + idle timeout for /ws/tts/multi ([#291](https://github.com/Kugelaudio/KugelAudio/issues/291)) ([d9d5e10](https://github.com/Kugelaudio/KugelAudio/commit/d9d5e106cb9d837a94f2bac813786ca78791c48a))

## [kugelaudio-python-sdk-v1.1.1](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.1.0...python-sdk-v1.1.1) (2026-04-18)

### Bug Fixes

* **python-sdk:** add in-flight timeout and release server context on barge-in ([#284](https://github.com/Kugelaudio/KugelAudio/issues/284)) ([d7ef023](https://github.com/Kugelaudio/KugelAudio/commit/d7ef023ff78b64329beac90c45971aa0e1a11705))

## [kugelaudio-python-sdk-v1.1.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v1.0.0...python-sdk-v1.1.0) (2026-04-18)

### Features

* **tts,web,python-sdk,java-sdk,js-sdk:** temperature parameter across stack + ISO 639-1 dictionary lookup ([#273](https://github.com/Kugelaudio/KugelAudio/issues/273)) ([439524f](https://github.com/Kugelaudio/KugelAudio/commit/439524f11f74bc251a58ddaa0015c90b49794ddc))

### Bug Fixes

* **python-sdk,tts:** multiplex TTS over single /ws/tts/multi connection ([#245](https://github.com/Kugelaudio/KugelAudio/issues/245)) ([1ca18a1](https://github.com/Kugelaudio/KugelAudio/commit/1ca18a1fb52ca2b8c37fed1e4999ac57ec790535))
* **tts,python-sdk,js-sdk,java-sdk:** keep WebSocket alive across streaming sessions ([#253](https://github.com/Kugelaudio/KugelAudio/issues/253)) ([2e22cc1](https://github.com/Kugelaudio/KugelAudio/commit/2e22cc1378e82e55c7889907318623da48515861))
* **tts,python-sdk,js-sdk,java-sdk:** remove redundant is_final signal from multi-context protocol ([#279](https://github.com/Kugelaudio/KugelAudio/issues/279)) ([3ee0bfa](https://github.com/Kugelaudio/KugelAudio/commit/3ee0bfab5867567280ad5cb4a8f70631574cbbf8))

## [kugelaudio-python-sdk-v1.0.0](https://github.com/Kugelaudio/KugelAudio/compare/python-sdk-v0.5.0...python-sdk-v1.0.0) (2026-04-15)

### ⚠ BREAKING CHANGES

* **python-sdk:** voices.list() now returns a VoiceListResponse object
instead of a plain list. Access the voice array via .voices.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
* **python-sdk,js-sdk,java-sdk:** voices.list() now returns a VoiceListResponse object
instead of a plain list. Access the voice array via .voices.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

### Features

* multi-region routing for all SDKs ([#227](https://github.com/Kugelaudio/KugelAudio/issues/227)) ([097efe2](https://github.com/Kugelaudio/KugelAudio/commit/097efe24ffdeb8814d066fc78b3b570ea1626e0f))
* **onboarding:** enhance onboarding logic to prevent orphan records ([d85fe05](https://github.com/Kugelaudio/KugelAudio/commit/d85fe054728d0cf1de0f055522b222074a12ddc7))
* **python-sdk,js-sdk,java-sdk:** add pagination support to voices list ([ad358ac](https://github.com/Kugelaudio/KugelAudio/commit/ad358ac51b2b2ada486d57f3ecce55e592f037c8))
* **python-sdk:** add pagination support to voices list ([37a2daa](https://github.com/Kugelaudio/KugelAudio/commit/37a2daa8ccb80cf9e2926652b198f707bb7b469d))
* **sdk:** add multi-region to LiveKit/Pipecat plugins, tests, and docs ([#239](https://github.com/Kugelaudio/KugelAudio/issues/239)) ([a45d000](https://github.com/Kugelaudio/KugelAudio/commit/a45d000876af4bfbba635af1a9678689c19293d0))
* **tts:** add support for multiple languages and validation ([8bb9978](https://github.com/Kugelaudio/KugelAudio/commit/8bb9978f359d0f5e0e87b127e33f2061700e79b7))

### Bug Fixes

* **tts:** correct positional arg mismatch in v1 streaming path ([b3d3746](https://github.com/Kugelaudio/KugelAudio/commit/b3d374637ad8d1a91967148f09f0f432280eea2a))
* **web,python-sdk:** show proper error when credits are exhausted (KUG-99) ([ce326c5](https://github.com/Kugelaudio/KugelAudio/commit/ce326c59e853d492ff07f362310aa4f0918976dc))

## [kugelaudio-python-sdk-v0.5.0](https://github.com/kajode/KugelAudio/compare/python-sdk-v0.4.0...python-sdk-v0.5.0) (2026-03-27)

### Features

* **python-sdk:** document 14 new supported languages in docstrings ([41dfc53](https://github.com/kajode/KugelAudio/commit/41dfc538b047d2936920c728f3289ca20f46dae0))

### Bug Fixes

* **js-sdk:** await session_closed before tearing down WebSocket on close (KUG-264) ([9a2a0b8](https://github.com/kajode/KugelAudio/commit/9a2a0b8a144485ed09fbf84cf9fa6a0bc7589811))
* **python-sdk:** handle 204 No Content responses in _request method ([5ab4561](https://github.com/kajode/KugelAudio/commit/5ab4561f4155ea4cd2b97ac366634a2c0eb5bb87))

## [kugelaudio-python-sdk-v0.4.0](https://github.com/kajode/KugelAudio/compare/python-sdk-v0.3.1...python-sdk-v0.4.0) (2026-03-24)

### Features

* **python-sdk:** add kugel-2 and kugel-2-turbo model IDs to pipecat TTSModels ([115873d](https://github.com/kajode/KugelAudio/commit/115873d221a47fc262f8a70344c1c9fc385cb2e6))
* **python-sdk:** add language parameter to LiveKit TTS plugin ([5586959](https://github.com/kajode/KugelAudio/commit/5586959501405c849a9ca6641c42653a6d4cd51b))
* **python-sdk:** add pipecat local bot example with mic/speaker transport ([32b248f](https://github.com/kajode/KugelAudio/commit/32b248f5203b4e4bd201ab2d52f5039f9c674ebf))
* **tts:** add chunk_length_schedule and auto_mode for server-side auto-chunking ([11ae4f0](https://github.com/kajode/KugelAudio/commit/11ae4f025a7df22bf0876e6e99e0b7da9d3f3703))
* **tts:** add playback speed control parameter to API ([e039978](https://github.com/kajode/KugelAudio/commit/e0399780ff896400d64d3a18ae3525cf800b544c))
* **tts:** update text normalization configurations and enhance streaming tests ([a3399ad](https://github.com/kajode/KugelAudio/commit/a3399adca7b1f97059d0b64a951f4f8b4d467a2e))

### Bug Fixes

* **infra:** persist block volume in fstab on NFS fallback ([ca3c607](https://github.com/kajode/KugelAudio/commit/ca3c607dda0cd52ffd5c0ad6670e244996dac445))
* **python-sdk:** fix typing, resource leaks, and code quality issues ([4cfd68a](https://github.com/kajode/KugelAudio/commit/4cfd68aa8b6b18233f19f4545be5d2941c84a8e7))
* **python-sdk:** wait for chunk_complete signal instead of short-polling after audio starts ([81f9283](https://github.com/kajode/KugelAudio/commit/81f92832352569c7a93c23bd5e231c273585cd9f))

## [kugelaudio-python-sdk-v0.3.1](https://github.com/kajode/KugelAudio/compare/python-sdk-v0.3.0...python-sdk-v0.3.1) (2026-03-02)

### Code Refactoring

* **ci:** use semantic-release-monorepo for path-based SDK commit filtering ([d161c5c](https://github.com/kajode/KugelAudio/commit/d161c5cf402d240ce421095833af7b8dfc8a9127))

## [0.3.0](https://github.com/kajode/KugelAudio/compare/python-sdk-v0.2.3...python-sdk-v0.3.0) (2026-03-02)

### Features

* add continuous streaming training support and audio segment ([54d5799](https://github.com/kajode/KugelAudio/commit/54d57992822803af54eed8e482e414077cbe0344))
* **asr/benchmarking:** add entity-level precision/recall metrics, NeMo engine wrappers, and pipeline CLI ([257831e](https://github.com/kajode/KugelAudio/commit/257831e5e8de68f196d81f75ec30d6dae8381be0))
* **asr/confidence:** add multi-signal confidence scoring (acoustic, phonetic, structural, contextual, cross-entity) ([282d015](https://github.com/kajode/KugelAudio/commit/282d0156c2cb2922582cd279b7989fefbde870e5))
* **asr/inference:** upgrade to Parakeet-TDT-0.6b-v3 with beam search, phrase boosting, and timestamp config ([bab8d21](https://github.com/kajode/KugelAudio/commit/bab8d218a7f15574740f7660a089fbb50f6f8f6e))
* **asr/itn:** add inverse text normalisation engine with LLM correction and multi-pass grammar ([e6be3ba](https://github.com/kajode/KugelAudio/commit/e6be3bad42ca608b3591c81a51a0f1482b8e0431))
* **asr/pipeline:** add end-to-end ASR orchestrator and Ray Serve serving layer ([f455905](https://github.com/kajode/KugelAudio/commit/f455905f308a3cee341d0fca9fefd028e85b9199))
* **asr/preprocessing:** add audio preprocessing pipeline (denoiser, normalizer, resampler, codec detector) ([82ffb93](https://github.com/kajode/KugelAudio/commit/82ffb932884d5bc3df1ac1f17c1586741e910fe6))
* **asr/training:** add Parakeet-TDT fine-tuning script and data generation scripts ([b9acd6b](https://github.com/kajode/KugelAudio/commit/b9acd6b161aa6604d9ee97d9918e4d9e5a5b78bc))
* **asr:** add entity extraction, phonetic disambiguation, vocabulary store, spelling detection, and reference data loader ([4d80722](https://github.com/kajode/KugelAudio/commit/4d8072216d13fe686030aaa3aee857e03f26530e))
* **asr:** add new ASR module with training infrastructure ([a124530](https://github.com/kajode/KugelAudio/commit/a124530f3b6bba1a3800a3d341e97edfb3331335))
* **asr:** add ResolutionStatus, shared LLM base, ITN DRY fixes, and extractor integration ([fa06250](https://github.com/kajode/KugelAudio/commit/fa062507d07dd1ad1fbd560134d266869a6aeb34))
* **billing:** update credit usage calculations and metrics ([89c94b0](https://github.com/kajode/KugelAudio/commit/89c94b09f08eb98afe3b9caf38b451c4d1e19843))
* **ci:** add platform input option for building self-hosted images ([e8de399](https://github.com/kajode/KugelAudio/commit/e8de39992f27463919e0e9ea0f53e830af9d32e8))
* **docs:** enhance documentation with cURL examples and Mintlify integration ([3edb931](https://github.com/kajode/KugelAudio/commit/3edb931eb2f59a5e373caff5398c000055409fbc))
* **docs:** update TTS API documentation with new parameters and limits ([d8891a5](https://github.com/kajode/KugelAudio/commit/d8891a5b4c4832162da3b03921a2bb5c25918099))
* **infra:** add license server, deepfilternet package, self-hosted docker-compose, and dependency updates ([3bdcb42](https://github.com/kajode/KugelAudio/commit/3bdcb4272975eb32a90ada63921f5307985e78e0))
* **java-sdk:** initial release of KugelAudio Java SDK on Maven Central ([b6139d5](https://github.com/kajode/KugelAudio/commit/b6139d5e99947f35355f0b2f7186861e18995e95))
* **java-sdk:** publish official Java SDK to Maven Central ([facfc51](https://github.com/kajode/KugelAudio/commit/facfc515ed13206d765a085b928e8899a81982cb))
* **license-server:** enhance configuration management, add R2 support, and improve activation logic ([c55b846](https://github.com/kajode/KugelAudio/commit/c55b846f0d864e55b1b53639efe0a625ccb38306))
* **localization:** add onboarding referral messages in multiple languages ([850bd0a](https://github.com/kajode/KugelAudio/commit/850bd0abf2c514f191f5ef67d363aef54013319a))
* **monitoring:** add Prometheus alerting rules and Grafana dashboards for TTS ([1fe1cbd](https://github.com/kajode/KugelAudio/commit/1fe1cbdf173b6344afb33644c8aa913441419b32))
* **monitoring:** enhance alerting rules and dashboards for Ray Serve metrics ([16e6418](https://github.com/kajode/KugelAudio/commit/16e6418602e396ade7fb9127c3b98c9f74fc5b47))
* **monitoring:** refine alerting rules and enhance usage metrics for TTS ([dd572b0](https://github.com/kajode/KugelAudio/commit/dd572b0d697720ccaaea9d034556613d0eb3aa93))
* New Landing Page with Animations, Edge Cases Demo & Full i18n ([82e3177](https://github.com/kajode/KugelAudio/commit/82e3177e5464858a69e138043d1cb79868fbcfc0))
* new landing page with scroll animations, edge cases demo, pricing calculator & full i18n ([ea6a454](https://github.com/kajode/KugelAudio/commit/ea6a454aac118a1474380170fd04c70d5daf3855)), closes [#pricing-calculator](https://github.com/kajode/KugelAudio/issues/pricing-calculator) [#languages](https://github.com/kajode/KugelAudio/issues/languages)
* **python-sdk:** add normalize support to LiveKit integration (default true) ([6f9fe78](https://github.com/kajode/KugelAudio/commit/6f9fe78d515fdb4ff3df100343178f243d135a6e))
* **python-sdk:** add normalize support to LiveKit integration (default true) ([#44](https://github.com/kajode/KugelAudio/issues/44)) ([7021e08](https://github.com/kajode/KugelAudio/commit/7021e08ad4b6ea76be08177166735e1cee7233fa))
* **release:** enhance release workflows and commit conventions ([8d4d49a](https://github.com/kajode/KugelAudio/commit/8d4d49a8649b1409c72c6019f932697adaaebb4b))
* **tts/bench,web:** add RTF benchmark mode and self-hosted license management UI ([2812ae8](https://github.com/kajode/KugelAudio/commit/2812ae81ad7f39bf60255714be4325d2be2aa7a8))
* **tts/consistency:** update default init_timestep to 999 and add v37-v44 configs ([c30de24](https://github.com/kajode/KugelAudio/commit/c30de24a70c41ffc5fe3710da1d683dd3dbc38e9))
* **models/tts/data:** add language detection, metadata backfill, dataset splitting utilities and SLURM jobs ([0856a9f](https://github.com/kajode/KugelAudio/commit/0856a9f0d19bcac292e94ea49efd09e3423b320b))
* **models/tts/data:** add unified dataset format with conversion pipeline and SLURM jobs ([508f236](https://github.com/kajode/KugelAudio/commit/508f2363e20410c56137c426848da5f818152378))
* **models/tts/data:** extend YouTube processing, unification CLI, and add SLURM batch jobs ([07e99ad](https://github.com/kajode/KugelAudio/commit/07e99ad573d88bc91740da601a423a0e043f1859))
* **tts/licensing:** add self-hosted licensing module with DRM, watermarking, and offline token support ([464b980](https://github.com/kajode/KugelAudio/commit/464b980e59a7d4599fd63a6104325cafb1be772b))
* **models/tts/scripts:** add unified distillation training script and SLURM job ([a84a454](https://github.com/kajode/KugelAudio/commit/a84a454649e84b0ec4e02a3cdc4e2f155977e222))
* **tts/selfhosted:** add self-hosted Docker image, weight protection build script, and CI workflow ([707bdf3](https://github.com/kajode/KugelAudio/commit/707bdf31939db2f9120d896f106ea10e9b6e1127))
* **tts/serving:** integrate self-hosted mode into serving layer (auth bypass, watermarking, license guard, diagnostics) ([78fc629](https://github.com/kajode/KugelAudio/commit/78fc62918135a88e82f6263c2cce97ad1516a90a))
* **tts:** add A8W8 static PTQ calibration and inference support ([ff234a6](https://github.com/kajode/KugelAudio/commit/ff234a65b4d2323959fe08eacf7e8d3f96befc53))
* **tts:** add arm64 self-hosted image build pipeline ([01bf5e3](https://github.com/kajode/KugelAudio/commit/01bf5e37b234b16b1e3785d69a291508d2e92352))
* **tts:** add continuation break injection, DMD2 guidance updates, and new engine presets ([8de6669](https://github.com/kajode/KugelAudio/commit/8de66699bfb13be16506cf726486426b1c07346f))
* **tts:** add INT4 weight-only quantization via TorchAO ([7ae4f49](https://github.com/kajode/KugelAudio/commit/7ae4f49c3196fdde4a8911114407e1f827dc41b9))
* **tts:** add keepalive ping interval configuration for WebSocket connections ([8dbe9ff](https://github.com/kajode/KugelAudio/commit/8dbe9ff7059d3b66c90e452f6d2fc029a73931db))
* **tts:** add optimization argument for model loading and inference ([ed65b91](https://github.com/kajode/KugelAudio/commit/ed65b91311b9e3d14e05ab2de72cd64a0f52b85e))
* **tts:** add torch/safetensors deps and harden watermark in license server ([880378e](https://github.com/kajode/KugelAudio/commit/880378ee1d32e5bc4a0ec74a2c1b5064741be976))
* **tts:** add tqdm progress logs and startup pending-license provisioning ([17b47e9](https://github.com/kajode/KugelAudio/commit/17b47e9765c35319f17553f827a2b0a04ae1ad54))
* **tts:** add Vapi custom TTS endpoint and ElevenLabs WebSocket stream-input ([725a61f](https://github.com/kajode/KugelAudio/commit/725a61fa92640734696e2f8b073cdecd57fed145))
* **tts:** DMD on-the-fly teacher sampling + full audio validation ([62b06b5](https://github.com/kajode/KugelAudio/commit/62b06b57cb85b0b6f69734aadc3055176e0b0dc6))
* **tts:** enable voice access for self-hosted containers via license server ([ede15ed](https://github.com/kajode/KugelAudio/commit/ede15edf3428fbf8444315bb63f9822a89ba32d8))
* **tts:** enhance audio processing and model configuration ([ec784b1](https://github.com/kajode/KugelAudio/commit/ec784b14b57fc641b005b2f77ace6ec26400a043))
* **tts:** implement voice retrieval and response handling for organizations ([f222db3](https://github.com/kajode/KugelAudio/commit/f222db371769e143f20197312d5807823c8d0e3d))
* **tts:** integrate weight provisioning into license server ([92d63d8](https://github.com/kajode/KugelAudio/commit/92d63d88677d4a20d715690e464e9d1f6ba03878))
* **tts:** merge new-model-train — continuation breaks, DMD2, engine presets ([7f705be](https://github.com/kajode/KugelAudio/commit/7f705be5ba6aa47370cc1c4240a0900a02a1d9e1))
* **tts:** refine audio processing and model training configurations ([d82469e](https://github.com/kajode/KugelAudio/commit/d82469e304e12e01acd1356194bbe0ee1e549a98))
* **tts:** stream-merge sharded HF weights without loading into RAM ([3cf93ae](https://github.com/kajode/KugelAudio/commit/3cf93ae1482b27ec46bd0032ea37ec8205981db6))
* **tts:** update optimization configurations and add new packages ([32d7dd0](https://github.com/kajode/KugelAudio/commit/32d7dd07b10006723be4412307d694171de50e52))
* **types:** add new dataclasses for Activation, License, and UsageEvent column types ([37cce34](https://github.com/kajode/KugelAudio/commit/37cce349011531f8df95064034cc93f36528cbb7))
* **web:** add language support to TTS generation ([226f57a](https://github.com/kajode/KugelAudio/commit/226f57a731584d832d793135e933d0e14acb6f4a))
* **web:** add provision weights UI with SSE progress streaming ([7a10dd4](https://github.com/kajode/KugelAudio/commit/7a10dd46291e341b8678b0a0d7bfac3f845c72cd))
* **web:** add subscription management to admin organizations page ([cd35fcd](https://github.com/kajode/KugelAudio/commit/cd35fcd6b642188bcb0d759799e9a422b095bf8b))
* **web:** add TTFA latency tracking and visualization ([26c22e6](https://github.com/kajode/KugelAudio/commit/26c22e6188fd44ce214165b823ae888407c1f9e3))
* **web:** enhance audio playback handling across components ([4ee8333](https://github.com/kajode/KugelAudio/commit/4ee833397bae68f1986d791f87054446444cc934))

### Bug Fixes

* **ci:** add Docker Hub login to deploy validation, normalize v-prefix ([0c84ffc](https://github.com/kajode/KugelAudio/commit/0c84ffccb57967a92fdf5c47218a00fa6980fe50))
* **ci:** correct smoke test to detect Nuitka-compiled modules via loader type ([9a0ff66](https://github.com/kajode/KugelAudio/commit/9a0ff66d71acfd0f98da396ed0f5e451dd83ec61))
* **ci:** correctly apply selfhosted config in release workflow ([1f697a5](https://github.com/kajode/KugelAudio/commit/1f697a5f35aefbd9c5b0f5442df2645f806c9927))
* **ci:** fix ffmpeg apt install in ARM Dockerfile via policy-rc.d ([709c8bc](https://github.com/kajode/KugelAudio/commit/709c8bcce6b8923627618a3fb0d6cbdfa43b4e1f))
* **ci:** fix staging teardown to actually trigger ArgoCD pruning ([3a0351a](https://github.com/kajode/KugelAudio/commit/3a0351a2e29bc5ad154d4ff374e28da08342f19e))
* **ci:** remove scope requirement from SDK release configs ([febf308](https://github.com/kajode/KugelAudio/commit/febf3086aa81eb2aef4833f334c9b724c61ef7b4))
* **ci:** replace docker image prune -a with targeted kugelaudio/pytorch image removal ([3b05ef7](https://github.com/kajode/KugelAudio/commit/3b05ef72bfc11b5f414d74a4072d016fe5798171))
* **ci:** seed dockerhub-creds and tts-credentials into staging namespace on deploy ([5fefa02](https://github.com/kajode/KugelAudio/commit/5fefa02fbd10e1d3b21c5e387fdcafe8f46df423))
* **ci:** stub invoke-rc.d alongside update-rc.d to fix x11-common postinst ([2c0bc21](https://github.com/kajode/KugelAudio/commit/2c0bc217ab27a546f5560fec11d1421db0c39cd5))
* **ci:** stub update-rc.d to fix x11-common post-install in container ([e6a081a](https://github.com/kajode/KugelAudio/commit/e6a081af6864c0054bfc05ae0200fcfb94a4baf4))
* **ci:** use adduser/addgroup instead of useradd for Debian 13 base ([4a8f46a](https://github.com/kajode/KugelAudio/commit/4a8f46a827449df5620186591d3094f5e087e3cd))
* **ci:** use config swap instead of extends for SDK semantic-release ([6bfe8b6](https://github.com/kajode/KugelAudio/commit/6bfe8b66a0b2b6c2849fa452d5c03863be49256a))
* **ci:** use double-quoted printf for policy-rc.d ([add3fa3](https://github.com/kajode/KugelAudio/commit/add3fa3a021e26250f33a6832adbe6a7ed5f606b))
* **ci:** use printf for policy-rc.d to correctly write newline ([561af8e](https://github.com/kajode/KugelAudio/commit/561af8edd43fc0b83d53f65fb926de46669bc534))
* **imprint:** update company name and contact details in imprint page ([b6be77a](https://github.com/kajode/KugelAudio/commit/b6be77a53f4c47c64a16fa7c7ae8a0937136cd25))
* **infra:** set staging worker CPU/memory limits to match A100 node capacity ([b5d6d4d](https://github.com/kajode/KugelAudio/commit/b5d6d4d5d59ae71f4f44c576f3aaf86d42a4a8dc))
* **java-sdk:** disable javadoc doclint to allow HTML heading flexibility in docs ([c493161](https://github.com/kajode/KugelAudio/commit/c4931614605d755f11484776644aa3336712bbd6))
* **js-sdk:** add toReadable() to eliminate audio gaps in Vapi streaming ([d62a75a](https://github.com/kajode/KugelAudio/commit/d62a75afaa8d9c5a8a64c8bb6dbf44f3797ae248))
* **js-sdk:** add toReadable() to eliminate audio gaps in Vapi streaming ([#46](https://github.com/kajode/KugelAudio/issues/46)) ([798f18f](https://github.com/kajode/KugelAudio/commit/798f18f47833964e129b6bf213a8aa785d5c8b65))
* miscellaneous fixes and lockfile update ([7910737](https://github.com/kajode/KugelAudio/commit/79107379db9024326e34e4b5e0a89274df7c6543))
* **privacy, terms:** update company name from Kugelaudio UG to KugelAudio GmbH ([af280b2](https://github.com/kajode/KugelAudio/commit/af280b2e56b44443e7911988959d93250fe9996b))
* resolve OAuth redirect to localhost:3000 in production ([c4406dd](https://github.com/kajode/KugelAudio/commit/c4406ddb16b6cc72cd34db858c7b5c77bc8a0e67))
* **selfhosted:** preserve cached weight file across container restarts ([dda07f7](https://github.com/kajode/KugelAudio/commit/dda07f739b66e88261fbfbe360a9be1d25ac05f3))
* **selfhosted:** prevent stale weights cache after interrupted container start ([f9c1642](https://github.com/kajode/KugelAudio/commit/f9c164229772b0798b92153bbcfced6d78355810))
* **selfhosted:** replace cycjimmy action with direct npx semrel call ([43457ab](https://github.com/kajode/KugelAudio/commit/43457ab0097f882360368092a365614367b16ff4))
* **selfhosted:** use negated scope pattern to prevent catch-all suppression rules from overriding patch releases ([850ed76](https://github.com/kajode/KugelAudio/commit/850ed76d9ff6d7cb0845b3f853dad8d983c81f57))
* **tts:** add python-multipart dependency for FastAPI form data support ([73bcb9e](https://github.com/kajode/KugelAudio/commit/73bcb9e246014b36b252058f0212d6f56a13d009))
* **tts:** align flashinfer-jit-cache to 0.6.4 to match flashinfer-python and flashinfer-cubin ([064f036](https://github.com/kajode/KugelAudio/commit/064f03664b516d8b283478e56a730a77ee64c68d))
* **tts:** bypass torch.multinomial 2^24 category limit in weighted sampler ([07df03a](https://github.com/kajode/KugelAudio/commit/07df03a069c0dfaced0463d5b45a45d9f0319d0b))
* **tts:** expose _head/_scheduler proxies on CompiledDiffusionEngine ([aa93ae6](https://github.com/kajode/KugelAudio/commit/aa93ae6684a94c9ef503ffc8bd7e2281f3dbba03))
* **tts:** fail fast with clear error when python-multipart is missing ([9a4bbe4](https://github.com/kajode/KugelAudio/commit/9a4bbe426011a28778c05dba0bcf1f7018828d10))
* **tts:** fix numpy version conflict blocking Docker build ([68ffac3](https://github.com/kajode/KugelAudio/commit/68ffac39cf8149b40f3edf9ef9a2fa3242c495ec))
* **tts:** include is_public and project_id in get_voice_by_id select ([845e318](https://github.com/kajode/KugelAudio/commit/845e318d9cea8ddefd80da07f13849311f47d30b))
* **tts:** install passwd pkg for useradd/groupadd on arm base ([5fc5e84](https://github.com/kajode/KugelAudio/commit/5fc5e84cdd678b5f304e713031fd39f6301d5d1b))
* **tts:** normalize currency symbols correctly in text preprocessing ([#45](https://github.com/kajode/KugelAudio/issues/45)) ([0d2444d](https://github.com/kajode/KugelAudio/commit/0d2444df459e69ed2dbf29d8b0b59c8ffa7dafac))
* **tts:** pass step= to experiment.log() so W&B audio samples appear at correct step ([d1f51c7](https://github.com/kajode/KugelAudio/commit/d1f51c72a2ba1d13b11b4d67444da1cecb413ccf))
* **tts:** propagate null_condition/scheduler into DMDStrategy and gate fm_loss ([22dbbb5](https://github.com/kajode/KugelAudio/commit/22dbbb54c9116eef5c2fdceb11a1b893df21da73))
* **tts:** redirect HF cache to /app/model_cache and create home dir ([7d3d215](https://github.com/kajode/KugelAudio/commit/7d3d215993463fa6f9d4d8a5818c142ba69e24e7))
* **tts:** remove silence padding between continuation turns ([50b0396](https://github.com/kajode/KugelAudio/commit/50b03961b0457b8adb8e5b5e5680fc6915e22a0f))
* **tts:** strip diffusion_head prefix from EMA weights in audio callback ([fd2e02e](https://github.com/kajode/KugelAudio/commit/fd2e02e86c98e6cb6218ced4c0de399a30cb74dc))
* **tts:** strip diffusion_head prefix from student weights in audio callback ([41bcb71](https://github.com/kajode/KugelAudio/commit/41bcb7158125ae5e1a9a6c7d9dbf20e9204d9979))
* **tts:** update German email test assertions to match lowercase connector words ([9eaf6df](https://github.com/kajode/KugelAudio/commit/9eaf6dfc434ce585b4c74426e97ee596a7a9a994))
* **tts:** update INT4 quantization implementation and validation frequency settings ([fafb8a7](https://github.com/kajode/KugelAudio/commit/fafb8a7025ce0d86f37fcba5ec861bb645227518))
* **tts:** update training configurations and enhance logging ([b9db734](https://github.com/kajode/KugelAudio/commit/b9db7345dee10f7cc88eccd6af392c1b2280f6e6))
* **tts:** use compile_mode=default for diffusion head in deep_cache preset ([b4652e1](https://github.com/kajode/KugelAudio/commit/b4652e1c4b5fd068556057cdd6607c60b661b9a2))
* update ArgoCD login URL and enhance TTS instructions ([c959eb8](https://github.com/kajode/KugelAudio/commit/c959eb88c7b713cca618bb38b50d0a625f96f416))
* **web:** add curl to license server image for health checks ([354ab56](https://github.com/kajode/KugelAudio/commit/354ab5692d8d9ca8c0606fe77e49241abd0bda93))
* **web:** copy kugel_supabase local dependency in license server Dockerfile ([c1598f2](https://github.com/kajode/KugelAudio/commit/c1598f259ba13762b820e20806c04c332ec2c548))
* **web:** decode base64-encoded PEM keys in license server config ([3580533](https://github.com/kajode/KugelAudio/commit/358053310f782ade6e7f5d89ed57c61bf6cdca8b))
* **web:** fix Nuitka crown-jewel .so not replacing .py in self-hosted image ([2a2e8ad](https://github.com/kajode/KugelAudio/commit/2a2e8ad91e9530d001894c824150310d91d423f5))
* **web:** make org picker search all orgs via server-side filtering ([d4178bf](https://github.com/kajode/KugelAudio/commit/d4178bf10a226be3004fc557dca8b9379ad60634))
* **web:** regenerate package-lock.json to fix npm ci sync error ([9419bbb](https://github.com/kajode/KugelAudio/commit/9419bbb4546ac643e8f31b5cacf73af779735c38))
* **web:** regenerate package-lock.json to include missing @swc/helpers ([cd33d52](https://github.com/kajode/KugelAudio/commit/cd33d5295a1b73712ba3b7484eaea1940f87602d))
* **web:** resolve all TypeScript CI type-check failures ([a583a49](https://github.com/kajode/KugelAudio/commit/a583a49100010398c6847f8b879e650151498e86))
* **web:** update SEO description and remove hero background image ([d0079af](https://github.com/kajode/KugelAudio/commit/d0079af9143fd61fb98b052fd3a2c3aff96a48d4))

### Performance Improvements

* **infra:** enable RAY_SERVE_THROUGHPUT_OPTIMIZED on staging ([d9ab53e](https://github.com/kajode/KugelAudio/commit/d9ab53e585fabdbbaf75d8ce63670ccaa70dc304))
* **infra:** set RAY_SERVE_THROUGHPUT_OPTIMIZED as pod-level env var ([568359a](https://github.com/kajode/KugelAudio/commit/568359aeaecf7a0c0e2359255677382f46a4c1ad))
* **tts:** add network RTT instrumentation to WS ingress TTFA logging ([7b06229](https://github.com/kajode/KugelAudio/commit/7b06229a4fc75d57a1ee903d514214258dad268c))
* **tts:** add ttfa_ms tracking to all endpoints and auth billing ([cd43e50](https://github.com/kajode/KugelAudio/commit/cd43e50dc520e1da91a71b9d382076a0289e4f4a))
* **tts:** cache credit checks and fix TTFA measurement overhead ([720332f](https://github.com/kajode/KugelAudio/commit/720332ff3070be3242bbc196aca41648680b194d))
* **tts:** offload critical-path CPU work off event loop ([d2a4e05](https://github.com/kajode/KugelAudio/commit/d2a4e053bb73f1ee62d6e5b372175562bb279104))

### Code Refactoring

* **asr/data:** switch synthesis to local TTS engine with voice reference pool ([fbcf154](https://github.com/kajode/KugelAudio/commit/fbcf154578933cd18ac07e5342ddcec2e957918d))
* **asr/itn:** trim LLM ITN interface and pass registry to essentials ([163962f](https://github.com/kajode/KugelAudio/commit/163962fe46e15ef18f1d5e9722d674548dd178ea))
* **asr:** unify LLM correction and confidence scoring logic ([f801995](https://github.com/kajode/KugelAudio/commit/f801995b696090f3aa212c3474ce0e73ff3770d7))
* **ci:** consolidate and fix GitHub Actions workflows ([ecf4020](https://github.com/kajode/KugelAudio/commit/ecf40205ef64176ee4c914017071877bd95fc940))
* **ci:** consolidate SDK releases into single workflow, delete unused autoscaler ([59cb7f3](https://github.com/kajode/KugelAudio/commit/59cb7f3a6540cd7d9ca76d62fa265ff214c56377))
* **imprint:** restructure imprint page layout and enhance styling ([fba88aa](https://github.com/kajode/KugelAudio/commit/fba88aa3306bc1069d29e1ef4e2db64651d5dd2e))
* **java-sdk:** rename package from io.kugelaudio to com.kugelaudio ([9f802d2](https://github.com/kajode/KugelAudio/commit/9f802d23345a1b35a748e810c0a5daba3e31daad))
* **tts/distillation:** extract shared distillation infrastructure ([b5539f7](https://github.com/kajode/KugelAudio/commit/b5539f7a9edd87584da4df833995e1e6be1ac426))
* **tts:** centralize currency normalization in BaseNormalizer ([#47](https://github.com/kajode/KugelAudio/issues/47)) ([5084bd9](https://github.com/kajode/KugelAudio/commit/5084bd94ca760f408d085b1de47c28b6f95a2dc8))
* **tts:** improve audio watermarking process and update self-hosted configuration ([909eee8](https://github.com/kajode/KugelAudio/commit/909eee8801fa9cac9ad4e19dcef4b05c1df9598b))
* **tts:** remove server-reported ttfa_ms from WebSocket protocol ([99abe7e](https://github.com/kajode/KugelAudio/commit/99abe7e6d55d5f4122d475d223b91feb11afaec9))
* **tts:** replace bare excepts with typed handlers and add cross-language regression tests ([f63734f](https://github.com/kajode/KugelAudio/commit/f63734fc37e188f5e76694258117784dc6a7c31e))
* update Docker images and workflows to use kugelaudio namespace ([e122bc6](https://github.com/kajode/KugelAudio/commit/e122bc6ee4910b60570a90da981fd7a87d322349))

# Changelog

All notable changes to the KugelAudio Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-17

### Added
- Initial release of the KugelAudio Python SDK
- **Models API**: List available TTS models (`client.models.list()`)
- **Voices API**: List voices (`client.voices.list()`) and get voice details (`client.voices.get()`)
- **TTS Generation**: Generate complete audio (`client.tts.generate()`)
- **Streaming**: Real-time audio streaming via WebSocket (`client.tts.stream()`)
- **Async Support**: Full async/await support (`stream_async()`, `generate_async()`)
- **Streaming Sessions**: LLM integration for real-time TTS (`client.tts.streaming_session()`)
- **Audio Utilities**: Save to WAV, get duration, RTF calculation
- **Error Handling**: Typed exceptions for auth, rate limits, validation errors
- **Single URL Architecture**: Connect to TTS server directly for minimal latency
