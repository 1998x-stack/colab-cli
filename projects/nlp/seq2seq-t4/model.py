"""Seq2Seq LSTM encoder-decoder — Sutskever et al. 2014, scaled for Colab T4.

Paper: 4-layer LSTM, 1000 cells, 384M params.
Ours:  2-layer LSTM, 256 cells, ~5M params (T4-friendly).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, SOS, EOS, UNK = 0, 1, 2, 3


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source sequence.

        Args:
            src: [B, src_len] token ids

        Returns:
            hidden: [num_layers, B, hidden_dim]
            cell:   [num_layers, B, hidden_dim]
        """
        embedded = self.dropout(self.embed(src))
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, hidden: torch.Tensor,
                cell: torch.Tensor) -> torch.Tensor:
        """Decode target sequence (teacher-forced).

        Args:
            tgt:    [B, tgt_len] token ids
            hidden: [num_layers, B, hidden_dim]
            cell:   [num_layers, B, hidden_dim]

        Returns:
            logits: [B, tgt_len, vocab_size]
        """
        embedded = self.dropout(self.embed(tgt))
        outputs, _ = self.lstm(embedded, (hidden, cell))
        return self.fc(self.dropout(outputs))


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            src: [B, src_len]
            tgt: [B, tgt_len] (includes <SOS> at position 0)

        Returns:
            logits: [B, tgt_len, vocab_size]
        """
        hidden, cell = self.encoder(src)
        return self.decoder(tgt, hidden, cell)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 80) -> torch.Tensor:
        """Greedy decode a batch of source sentences.

        Returns:
            indices: [B, max_len] (truncated at first <EOS>)
        """
        self.eval()
        B = src.size(0)
        hidden, cell = self.encoder(src)

        # Start token for each sentence
        tgt = torch.full((B, 1), SOS, dtype=torch.long, device=self.device)
        outputs = []

        for _ in range(max_len):
            # Manual step: embed → lstm → fc (avoid double-calling decoder)
            embedded = self.decoder.dropout(self.decoder.embed(tgt[:, -1:]))
            lstm_out, (hidden, cell) = self.decoder.lstm(embedded, (hidden, cell))
            logits = self.decoder.fc(self.decoder.dropout(lstm_out))
            pred = logits[:, -1, :].argmax(-1, keepdim=True)
            outputs.append(pred)
            tgt = torch.cat([tgt, pred], dim=1)

            if (pred == EOS).all():
                break

        return torch.cat(outputs, dim=1)  # [B, max_len]

    @torch.no_grad()
    def beam_decode(self, src: torch.Tensor, beam_width: int = 5,
                    max_len: int = 80) -> list[list[int]]:
        """Beam search decode — one sentence at a time.

        Returns list of token-id lists (one best hypothesis per sentence),
        truncated at <EOS>.
        """
        self.eval()
        results = []

        for i in range(src.size(0)):
            single_src = src[i:i+1]  # [1, src_len]
            hidden, cell = self.encoder(single_src)

            # Beam state: (sequence, log_prob, hidden, cell)
            beams = [([SOS], 0.0, hidden, cell)]

            for _ in range(max_len):
                candidates = []
                for seq, score, h, c in beams:
                    if seq[-1] == EOS:
                        candidates.append((seq, score, h, c))
                        continue

                    token_t = torch.tensor([[seq[-1]]], device=self.device)
                    embedded = self.decoder.dropout(self.decoder.embed(token_t))
                    lstm_out, (new_h, new_c) = self.decoder.lstm(embedded, (h, c))
                    logits = self.decoder.fc(self.decoder.dropout(lstm_out))
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

                    topk = torch.topk(log_probs, beam_width)
                    for token_id, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                        candidates.append((seq + [token_id], score + lp, new_h, new_c))

                # Prune to top beam_width
                candidates.sort(key=lambda x: x[1] / len(x[0]), reverse=True)
                beams = candidates[:beam_width]

                if all(s[-1] == EOS for s, _, _, _ in beams):
                    break

            best = beams[0][0]
            # Truncate at first EOS
            eos_pos = len(best)
            for j, tok in enumerate(best):
                if tok == EOS:
                    eos_pos = j + 1
                    break
            results.append(best[1:eos_pos])  # strip SOS, include EOS

        return results


def build_seq2seq(src_vocab_size: int, tgt_vocab_size: int,
                  embed_dim: int = 256, hidden_dim: int = 512,
                  num_layers: int = 2, dropout: float = 0.3,
                  device: torch.device = None) -> Seq2Seq:
    """Factory: build encoder-decoder with paper-default hyperparams."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Paper uses uniform init (-0.08, 0.08)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.uniform_(p, -0.08, 0.08)

    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Seq2Seq: embed={embed_dim}, hidden={hidden_dim}, "
          f"layers={num_layers}, params={total:,}")

    return model
