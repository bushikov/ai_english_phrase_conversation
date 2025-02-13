# 英語フレーズを覚えるためのアプリ

## 使い方
1. 環境変数用ファイルの準備
```
$ cp .env.example .env
```
2. 環境変数の編集
    .envに必要な環境変数を編集
3. 英語フレーズ用ファイルの準備
```
$ cp file/phrases.tsv.example file/phrases.tsv
```
4. 覚えたい英語フレーズをfile/phrases.tsvに編集
5. 起動
```
$ docker compose run --rm app
```