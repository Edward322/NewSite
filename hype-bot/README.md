# hype-bot — торговый бот HYPEUSDT для Bybit

Проект в разработке. План и принятые решения — в [docs/PLAN.md](docs/PLAN.md).
Пошаговая инструкция по запуску для Windows появится на этапе 4.

## Текущий статус

| Этап | Статус |
|---|---|
| 0. План | готов |
| 1. Данные | готов ([итог](reports/STAGE1_DATA.md)) |
| 2. Бэктестер | готов, ждёт подтверждения ([итог](reports/STAGE2_BACKTESTER.md)) |
| 3. Исследование стратегий | — |
| 4. Торговый движок | — |
| 5. Демо | — |
| 6. Реальная торговля | — |

## Безопасность

- Ключи API хранятся только в файле `.env` (он исключён из git).
  Шаблон — `.env.example`.
- Ключ создаётся без права вывода средств и с привязкой к IP. Подробная
  инструкция будет добавлена на этапе 4.

## Данные (этап 1)

```
pip install -r requirements-dev.txt
python -m bot.cli data download   # история HYPEUSDT + контракты для проверки устойчивости
python -m bot.cli data check      # отчёт о качестве → reports/data_quality.md
python -m bot.cli data pack       # архив upload/hype-data.zip для передачи
python -m bot.cli data unpack upload/hype-data.zip   # распаковка с проверкой sha256
python -m pytest                  # тесты
```

API Bybit закрыт для ряда стран, включая страну облачной среды разработки,
поэтому история скачивается на компьютере пользователя: на Windows достаточно
запустить `windows\download_data.bat` — см. [инструкцию](docs/DATA_DOWNLOAD_WINDOWS.md).
