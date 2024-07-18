import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence  # これを追加
import torchvision
from torchvision import transforms
from gensim.models import KeyedVectors
from tqdm import tqdm

# Word2Vec形式のファイルをロード
word_vectors = KeyedVectors.load_word2vec_format('data/glove.42B.300d.word2vec.txt', binary=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_word_vector(word):
    try:
        return word_vectors[word]
    except KeyError:
        return np.zeros(word_vectors.vector_size)  # 未知の単語の場合、ゼロベクトルを使用

# 1. データローダーの作成
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

    def __getitem__(self, idx):
        """
        対応するidxのデータ（画像，質問，回答）を取得．

        Parameters
        ----------
        idx : int
            取得するデータのインデックス

        Returns
        -------
        image : torch.Tensor  (C, H, W)
            画像データ
        question : torch.Tensor  (seq_len, vec_size)
            質問文を単語ベクトルに変換したもの
        answers : torch.Tensor  (n_answers, vec_size) [optional]
            各回答を単語ベクトルに変換したもの
        mean_answer : torch.Tensor  (vec_size) [optional]
            各回答の平均ベクトル
        """
        # 画像データの処理
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)

        # 質問文を単語ベクトルに変換
        question_words = process_text(self.df["question"][idx]).split(" ")
        question_vectors = [get_word_vector(word) for word in question_words]
        question_vectors = np.array(question_vectors)
        question = torch.Tensor(question_vectors)

        if self.answer and 'answers' in self.df:
            # 回答を単語ベクトルに変換
            answers = [process_text(answer["answer"]) for answer in self.df["answers"][idx]]
            answers_vectors = [get_word_vector(answer) for answer in answers]
            answers_vectors = np.array(answers_vectors)
            mean_answer = np.mean(answers_vectors, axis=0)  # 各回答の平均ベクトルを計算

            return image, question, torch.Tensor(answers_vectors), torch.Tensor(mean_answer)

        else:
            return image, question

    def __len__(self):
        return len(self.df)


# 2. 評価指標の実装
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    """
    VQA評価指標を計算する関数
    コサイン類似度を用いて予測ベクトルと回答ベクトルの類似度を計算し、平均類似度を返す。

    Parameters
    ----------
    batch_pred : torch.Tensor
        モデルの予測出力 (batch_size, vec_size)
    batch_answers : torch.Tensor
        正解の回答ベクトル (batch_size, n_answers, vec_size)

    Returns
    -------
    float
        平均類似度
    """
    total_similarity = 0.0

    for pred, answers in zip(batch_pred, batch_answers):
        # 予測ベクトルと各回答ベクトルのコサイン類似度を計算
        similarities = F.cosine_similarity(pred.unsqueeze(0), answers, dim=1)
        # 各回答ベクトルとの類似度の平均を計算
        avg_similarity = similarities.mean().item()
        total_similarity += avg_similarity

    return total_similarity / len(batch_pred)


# 3. モデルのの実装
# ResNetを利用できるようにしておく
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64)
        self.layer2 = self._make_layer(block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 512)

    def _make_layer(self, block, blocks, out_channels, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])


class VQAModel(nn.Module):
    def __init__(self, word_vector_size: int, n_answer: int):
        super().__init__()
        self.resnet = ResNet18()
        
        # LSTMを使用して質問文のベクトルを処理
        self.text_encoder = nn.LSTM(input_size=word_vector_size, hidden_size=512, num_layers=1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, question):
        image_feature = self.resnet(image)  # 画像の特徴量

        # LSTMに入力するために質問文ベクトルをエンコード
        _, (hidden, _) = self.text_encoder(question)
        question_feature = hidden[-1]  # 最後の隠れ状態を特徴量として使用

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x


# 4. 学習の実装
import torch.nn.functional as F

# 1. train関数の定義
def train(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mean_answer in tqdm(dataloader, desc="Training"):  # tqdmでラップ
        image, question, answers, mean_answer = \
            image.to(device), question.to(device), answers.to(device), mean_answer.to(device)

        pred = model(image, question)
        loss = criterion(pred, mean_answer)  # 損失関数を変更

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += VQA_criterion(pred, answers)  # VQA accuracyを更新
        simple_acc += F.cosine_similarity(pred, mean_answer).mean().item()  # simple accuracyをコサイン類似度で評価

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


# 2. eval関数の定義
def eval(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for image, question, answers, mean_answer in tqdm(dataloader, desc="Evaluating"):  # tqdmでラップ
        image, question, answers, mean_answer = \
            image.to(device), question.to(device), answers.to(device), mean_answer.to(device)

        with torch.no_grad():
            pred = model(image, question)
            loss = criterion(pred, mean_answer)  # 損失関数を変更

        total_loss += loss.item()
        total_acc += VQA_criterion(pred, answers)  # VQA accuracyを更新
        simple_acc += F.cosine_similarity(pred, mean_answer).mean().item()  # simple accuracyをコサイン類似度で評価

    return total_loss / len(dataloader), total_acc / len(dataloader), simple_acc / len(dataloader), time.time() - start


# 3. collate_fn関数の定義
def collate_fn(batch):
    images, questions, answers, mean_answers = zip(*batch)

    # 質問文をパディング
    questions_padded = pad_sequence(questions, batch_first=True)
    
    # 回答もパディング
    answers_padded = pad_sequence(answers, batch_first=True)

    return torch.stack(images), questions_padded, answers_padded, torch.stack(mean_answers)


# 4. main関数の定義
def main():
    # deviceの設定
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataloader / model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = VQADataset(df_path="./data/train.json", image_dir="./data/train", transform=transform)
    test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./data/valid", transform=transform, answer=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=collate_fn, num_workers=12, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=12, pin_memory=True)


    model = VQAModel(word_vector_size=300, n_answer=300).to(device)  # word_vector_sizeを設定

    # optimizer / criterion
    num_epoch = 20
    criterion = nn.MSELoss()  # 損失関数をMSELossに変更
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # train model
    for epoch in range(num_epoch):
        train_loss, train_acc, train_simple_acc, train_time = train(model, train_loader, optimizer, criterion, device)
        print(f"【{epoch + 1}/{num_epoch}】\n"
              f"train time: {train_time:.2f} [s]\n"
              f"train loss: {train_loss:.4f}\n"
              f"train acc: {train_acc:.4f}\n"
              f"train simple acc: {train_simple_acc:.4f}")
    
    # モデルの保存
    torch.save(model.state_dict(), "model.pth")
    print("モデルを保存しました。")

    # 提出用ファイルの作成
    model.eval()
    submission = []
    for image, question in test_loader:
        image, question = image.to(device), question.to(device)
        with torch.no_grad():
            pred = model(image, question)
        submission.append(pred.cpu().numpy())

    torch.save(model.state_dict(), "model.pth")
    np.save("submission.npy", np.array(submission))

if __name__ == "__main__":
    main()