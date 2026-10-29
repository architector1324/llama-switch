# Llama Switch

![](./ref-dark.png)

Один дашборд и один OpenAI-совместимый эндпоинт для локальных движков: [llama.cpp](https://github.com/ggerganov/llama.cpp), [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) и [audio.cpp](https://github.com/0xShug0/audio.cpp). Несколько моделей работают одновременно, каждая на своём порту и своих картах.

## Установка

```bash
curl -L -o ~/.local/bin/llmsw https://github.com/architector1324/llama-switch/releases/latest/download/llmsw-linux-amd64
chmod +x ~/.local/bin/llmsw
```

На Windows скачайте `llmsw-windows-amd64.exe` из [Releases](https://github.com/architector1324/llama-switch/releases). Или соберите сами, нужен Go 1.26+:

```bash
git clone https://github.com/architector1324/llama-switch.git
cd llama-switch/src && CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o ~/.local/bin/llmsw .
```

Для работы: `llama-server` и движки тех секций, которыми пользуетесь. Опционально: `ffmpeg` (загрузка аудио не в WAV), `vulkaninfo` (PCI-адреса в выборе GPU).

## Использование

```bash
llmsw serve -f config.yaml --watch     # дашборд и API на http://localhost:11435
```

### Терминал

```bash
llmsw list                             # модели из конфига
llmsw ps                               # запущенные модели
llmsw ps gemma4                        # всё, что дашборд показывает об одной
llmsw stop gemma4                      # или: llmsw stop --all
```

`run` работает с моделью любого типа и сначала загружает её:

```bash
llmsw run gemma4                                         # сессия чата, /bye - выход
llmsw run gemma4 "Скажи привет тремя словами"
llmsw run gemma4 photo.png "Что на картинке?"            # картинки, аудио, видео
git diff | llmsw run qwen3.6:35b --system "Ты пишешь сообщения коммитов." "Напиши одно"
llmsw run z-image "рыжий кот на подоконнике" --size 1024x1024 > cat.png
llmsw run mage-flow-edit cat.png "сделай кота чёрным"
llmsw run qwen3-tts "Привет!" --voice Kore
llmsw run qwen3-asr meeting.mp3
cat ticket.txt | llmsw run clef:9b "Is this urgent?"
```

- Файлы, названные в промпте, уходят вместе с ним: `llm` получает картинки, аудио и видео, `sd` правит картинки. Текст из пайпа уходит как есть.
- Картинки и аудио пишутся в `-o`, в stdout в пайпе, иначе в `МОДЕЛЬ-ВРЕМЯ.ext`. Аудио проигрывается (`--no-play`). Блоки кода из ответа сохраняются так же (`--no-save`).
- Настройки дашборда - тоже флаги: `--gpu`, `--ctx`, `--parallel`, `--preserve-thinking`, `--size`, `--hires`, `--frames`, `--single`.
- Флаги ставятся где угодно после `run`. После `--` всё идёт в промпт.
- `--verbose` печатает тайминги в stderr, `llmsw run -h` показывает все флаги.
- Сервер ищется по `LLMSW_HOST`, по умолчанию `localhost:11435`.

Автодополнение для fish знает модели, флаги каждого типа, карты и размеры:

```fish
llmsw completion fish > ~/.config/fish/completions/llmsw.fish
```

### Дашборд

Откройте `http://localhost:11435`, выберите секцию и модель, выставьте GPU и контекст, нажмите **Load**. Изменение поля перезапускает модель с ним. Микрофон работает только на `localhost`: страница отдаётся по обычному http.

### Сервис

```bash
cp llama-switch.service ~/.config/systemd/user/
systemctl --user enable --now llama-switch
journalctl --user -u llama-switch -f
```

Юнит запускает `llmsw serve` на `~/.config/llama-switch/config.yaml` с `--watch`.

### Флаги сервера

| Флаг | По умолчанию | |
| --- | --- | --- |
| `-H`, `--host` | `localhost` | интерфейс для UI и движков |
| `-p`, `--port` | `11435` | порт UI. Движки получают свободный порт |
| `-c`, `--ctx` | `4096` | контекст по умолчанию, на слот |
| `-np`, `--parallel` | `1` | слотов по умолчанию |
| `-f`, `--config` | `config.yaml` | файл конфига |
| `-w`, `--watch` | выкл | перечитывать конфиг при изменении, останавливая все модели |

## Настройка

До шести секций: `llm`, `sd`, `tts`, `stt`, `music` и `decision`. У каждой модели кванты указывают на строки запуска:

```yaml
llm:
  qwen-3:
    q4_k_m: llama-server --host ${HOST} --port ${PORT} ${DEVICES} -c ${CTX} --parallel ${PARALLEL} -ngl 99 -m /path/to/qwen3-q4_k_m.gguf
    q8_k_xl: llama-server --host ${HOST} --port ${PORT} ${DEVICES} -c ${CTX} --parallel ${PARALLEL} -ngl 99 -m /path/to/qwen3-q8_k_xl.gguf
  gemma-3:
    q4_k_m: llama-server --host ${HOST} --port ${PORT} ${DEVICES} -c ${CTX} --parallel ${PARALLEL} -ngl 99 -m /path/to/gemma-3-q4_k_m.gguf --mmproj /path/to/gemma-3-mmproj.gguf

sd:
  z-image:
    q8_0: sd-server -l ${HOST} --listen-port ${PORT} ${DEVICES:--backend diffusion=Vulkan1,te=Vulkan0,vae=Vulkan0} -W ${WIDTH} -H ${HEIGHT} ${HIRES:--hires --hires-denoising-strength 0.5} --diffusion-model /path/to/z_image_turbo-Q8_0.gguf --llm /path/to/Qwen3-4B-Q8_0.gguf --vae /path/to/ae.safetensors --steps 8 --cfg-scale 1.0

tts:
  qwen3-tts:
    q8: tts-server --host ${HOST} --port ${PORT} --max-new ${FRAMES} --model /path/to/qwen-talker-Q8_0.gguf --codec /path/to/qwen-tokenizer-Q8_0.gguf

stt:
  nemotron-asr:
    f16: audiocpp-server --host ${HOST} --port ${PORT} ${DEVICES} --config /path/to/nemotron-asr.json --busy-timeout-ms 0

music:
  minimax-music3:
    q4: audiocpp-server --host ${HOST} --port ${PORT} ${DEVICES} --config /path/to/music.json --busy-timeout-ms 0

decision:
  clef:
    q8_0: llama-server --host ${HOST} --port ${PORT} ${DEVICES} -c ${CTX} --parallel ${PARALLEL} -ngl 99 -m /path/to/Clef-Flash-Q8_0.gguf
```

Первый квант - квант по умолчанию, он доступен по голому id (`qwen-3`). Остальные получают суффикс (`qwen-3-q8_k_xl`). Id глобальные. Дубликат в следующей секции пропускается.

| Плейсхолдер | Типы | Превращается в |
| --- | --- | --- |
| `${HOST}`, `${PORT}` | все | интерфейс и свободный порт |
| `${CTX}` | `llm`, `decision` | контекст на слот, умноженный на число слотов, для `-c` |
| `${PARALLEL}` | `llm`, `decision` | число слотов |
| `${DEVICES}` | все | выбранная карта: `-dev Vulkan1`, `--backend vulkan --device 1` |
| `${DEVICES:<flags>}` | все | то же, или `<flags>` при **All GPUs** и **Profile**. Для `sd` задаёт роли |
| `${PRESERVE_THINK}` | `llm` | `--reasoning-preserve` или `--no-reasoning-preserve` |
| `${FRAMES}` | `tts` | лимит кадров |
| `${WIDTH}`, `${HEIGHT}` | `sd` | размер картинки |
| `${HIRES}`, `${HIRES:<flags>}` | `sd` | `--hires` или `<flags>` при включённом Hires Fix, иначе ничего |

## API

Запрос с `model` сначала загружает модель. Запрос без модели идёт в запущенную модель этого типа.

| Эндпоинт | Секция |
| --- | --- |
| `/v1/models` | все |
| `/v1/chat/completions`, `/v1/completions`, `/v1/responses` | `llm` |
| `/v1/images/*`, `/sdapi/v1/*`, `/sdcpp/v1/*` | `sd` |
| `/v1/audio/speech`, `/v1/audio/voices` | `tts` |
| `/v1/audio/transcriptions`, `/v1/audio/translations` | `stt` или `llm` с ASR llama-server |
| `/v1/tasks/*` | `music` |
| `/v1/systemone` | `decision` |

| Управление | |
| --- | --- |
| `GET /api/status[?model=]` | запущенные экземпляры и настройки |
| `POST /api/start` | запустить модель |
| `POST /api/stop[?model=]` | остановить одну или все. В ответе - список |
| `POST /api/reset[?model=\|?kind=]` | настройки по умолчанию |
| `POST /api/multi` | `{"multi": false}` - одна модель за раз |
| `GET /api/logs[?model=]`, `POST /api/logs/clear[?model=]` | лог модели |
| `GET /api/devices` | карты |
| `GET /api/config` | конфиг |
