import cv2
import json
import subprocess
import time
import requests
import ollama
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog


# ── Запуск Ollama если не запущен ─────────────────────────────────────────────

def ensure_ollama():
    try:
        requests.get("http://localhost:11434", timeout=2)
    except:
        print("  Запускаю Ollama...")
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)


# ── Распознавание команды через локальную нейросеть ───────────────────────────

MODEL = "qwen2.5:3b"   # локальная модель через Ollama

def neural_network(command):
    prompt = f"""Ты — помощник по обработке изображений. Пользователь пишет команду на любом языке.
Твоя задача — определить что нужно сделать с изображением и вернуть ТОЛЬКО JSON объект.

Доступные действия:
- {{"action": "rotate", "angle": 90}}             — повернуть на любой угол в градусах
- {{"action": "resize", "width": 300, "height": 200}} — изменить размер
- {{"action": "channel", "color": "red"}}         — выделить канал: red / green / blue
- {{"action": "grayscale"}}                       — чёрно-белое
- {{"action": "edges"}}                           — найти границы (Canny)
- {{"action": "flip", "direction": "horizontal"}} — отразить: horizontal / vertical
- {{"action": "blur"}}                            — размыть
- {{"action": "unknown"}}                         — если команда непонятна

Команда пользователя: "{command}"

Верни ТОЛЬКО JSON, без пояснений и без markdown."""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        return json.loads(response["message"]["content"])
    except Exception as e:
        print(f"  Ошибка нейросети: {e}")
        return {"action": "unknown"}


# ── Выполнение OpenCV операции ────────────────────────────────────────────────

def execute_command(image, command):
    print("  Думаю...")
    action = neural_network(command)
    print(f"  Нейросеть решила: {action}")

    act = action.get("action", "unknown")

    if act == "rotate":
        angle = action.get("angle", 90)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
        result = cv2.warpAffine(image, M, (cols, rows))
        title = f"Поворот на {angle}°"

    elif act == "resize":
        w, h = action.get("width", 300), action.get("height", 300)
        result = cv2.resize(image, (w, h))
        title = f"Новый размер: {w} x {h}"

    elif act == "channel":
        B, G, R = cv2.split(image)
        color = action.get("color", "red")
        if color == "green":
            result, title = G, "Зелёный канал"
        elif color == "blue":
            result, title = B, "Синий канал"
        else:
            result, title = R, "Красный канал"

    elif act == "grayscale":
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        title = "Чёрно-белое"

    elif act == "edges":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, 100, 200)
        title = "Границы объектов"

    elif act == "flip":
        direction = action.get("direction", "horizontal")
        code = 0 if direction == "vertical" else 1
        result = cv2.flip(image, code)
        title = f'Отражение {"по вертикали" if direction == "vertical" else "по горизонтали"}'

    elif act == "blur":
        result = cv2.GaussianBlur(image, (15, 15), 0)
        title = "Размытие"

    else:
        print("  Не смог распознать команду.")
        return

    # Показываем оригинал и результат рядом
    plt.figure(figsize=(11, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Оригинал", fontsize=14)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if len(result.shape) == 2:
        plt.imshow(result, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=14)
    plt.axis("off")

    plt.suptitle(f'Команда: "{command}"', fontsize=12, color="gray")
    plt.tight_layout()
    plt.show()


# ── Запуск ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Управление изображением через нейронку")
    print(f"  Модель: {MODEL} (локально через Ollama)")
    print("=" * 50)
    print()

    ensure_ollama()

    root = tk.Tk()
    root.withdraw()

    # Выбор картинки через окно проводника
    print("Сейчас откроется окно — выбери картинку...")
    path = filedialog.askopenfilename(
        title="Выбери картинку",
        filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Все файлы", "*.*")]
    )

    if not path:
        print("Файл не выбран.")
        return

    image = cv2.imread(path)
    if image is None:
        print("Ошибка: не удалось открыть файл.")
        return

    print(f"Картинка загружена: {image.shape[1]} x {image.shape[0]} пикселей")
    print()
    print("Пиши команду на русском или английском — нейронка поймёт.")
    print()
    print("  Примеры команд:")
    print("  ┌─ поверни на 90 / на 45 / на 180")
    print("  ├─ измени размер до 400x300")
    print("  ├─ выдели красный / зелёный / синий канал")
    print("  ├─ сделай чёрно-белым")
    print("  ├─ найди границы")
    print("  ├─ отрази горизонтально / вертикально")
    print("  ├─ размой изображение")
    print("  ├─ смени картинку  — загрузить другой файл")
    print("  └─ выход           — закрыть программу")
    print("-" * 50)

    while True:
        print()
        command = input("Твоя команда: ").strip()

        if not command:
            continue

        if command.lower() in ["выход", "exit", "quit", "q"]:
            print("Пока!")
            root.destroy()
            break

        # Смена картинки без перезапуска
        if command.lower() in ["смени картинку", "сменить картинку", "смена картинки", "смена картинку", "другая картинка", "change image", "change picture"]:
            print("Сейчас откроется окно — выбери новую картинку...")
            root.deiconify()
            root.lift()
            root.focus_force()
            root.withdraw()
            new_path = filedialog.askopenfilename(
                title="Выбери картинку",
                filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Все файлы", "*.*")]
            )
            if new_path:
                new_image = cv2.imread(new_path)
                if new_image is not None:
                    image = new_image
                    print(f"Новая картинка загружена: {image.shape[1]} x {image.shape[0]} пикселей")
                else:
                    print("Не удалось открыть файл, картинка не изменена.")
            else:
                print("Файл не выбран, картинка не изменена.")
            continue

        execute_command(image, command)


if __name__ == "__main__":
    main()
