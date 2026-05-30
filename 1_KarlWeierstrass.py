"""
Ako diferenciraš beskonačnu sumu kosinusnih funkcija i pogledaš koeficijente ispred kosinusa, 
videćeš da njihova suma ne konvergira, a kad nema konvergencije, nema ni diferencijabilnosti.

Za krivu f(t) = lex-indeks to praktično znači:
Nema lokalne predvidljivosti smera — diferenciranje (ovde: inkrementi dX = f(t+1) - f(t)) ne daje stabilan "izvod". 
Drugim rečima, ne postoji broj koji opisuje "trenutnu brzinu" krive; ona u svakom koraku skoči i levo i desno za milione.
Moji rezultati to potvrđuju: Higuchi FD ≈ 1.9988 (skoro maksimalno hrapavo), varijansa scaling slope ≈ 0.0018 (Brown referenca = 1), inkrementi nisu normalni. 
Sve to kaže da kriva pripada Weierstrass-tipu — neprekidna ali bez prave tangente.
Zato i nije moguće iz jednog poteza pogoditi tačan sledeći lex-indeks; "izvod" ne postoji.
Ono što ipak postoji je globalna struktura: Hurst ≈ 0.59 (slaba perzistentnost), lag-1 ACF/MI nad inkrementima ≈ 0.5 (jedan korak unazad nosi informaciju), rolling H/FD memorija. 
To koriste sledeci NEXT modeli — ne kao izvod, nego kao slabe statističke "ručke" oko kojih grade region kandidata, ne jednu tačku.
Ukratko: kriva je "fraktalno-Brownova" po prirodi, lokalno nepredvidljiva (kao Weierstrass), 
globalno blago perzistentna — i to je upravo prozor kroz koji predikcije imaju ikakvog smisla.
Ne tražim "tangentnu formulu" za sledeći broj, nego male statističke tragove u inače veoma hrapavoj krivi.
Ne predviđam glatku krivu. Lovim ponavljajuće tragove u hrapavoj krivi.


KarlWeierstrass algorithm for Loto 7/39 prediction

1. Weierstrass-ova funkcija nad svih 4624 do sad izvucenih kombinacija.       

2. Aparati.  
    a) Brownovo kretanje.    
    b) Hurst eksponent (R/S analiza)   
    c) Fraktalna dimenzija.      

3. Nad svakim aparatom svaki Test  
    a) Hurst eksponent.    
    b) Autokorelacija (ACF).   
    c) Mutual information.   
    d) Sample / permutation entropy.   
    e) NIST baterija testova nasumičnosti.          

Strukturu testiram nad niz 4624 lex-indeksa (iz skalar 1..15.380.937).      
Kasnije izvrsiti analizu rezultata i izabrati sta je najbolje za predikciju sledece loto kombinacije


Koncept:
Niz nije 7 kolona nego jedan skalarni niz: svaka Loto 7/39 kombinacija se mapira u lex-indeks 1..C(39,7).
Nad tim nizom od 4624 lex-indeksa posmatram "Weierstrass/fraktalnu" krivu: 
možda nema tangente, ali možda ima skrivenu strukturu.
Cilj nije odmah predikcija, nego prvo analiza: da li postoji prediktivna struktura.
Aparati: Brownovo kretanje, Hurst/R-S, fraktalna dimenzija.
Testovi: Hurst, ACF, mutual information, sample/permutation entropy, NIST randomness.


Koristim:
Izvučene kombinacije: 4624
Ceo prostor kombinacija: 39C7 (15.380.937 kombinacija)
Svaka izvučena kombinacija se mapira na tačan red/indeks u 39C7 prostoru.
Dobijam niz od 4624 lex-indeksa: to su stvarne 4624 tačke Weierstrass-ove krive.
Ne radim krivu od svih 15.380.937 tačaka, nego krivu od 4624 stvarno posećene tačke, 
gde je svaka tačka njen indeks u ukupnom prostoru.
Dakle f(t) je: t = redni broj izvlačenja 1..4624 f(t) = indeks te kombinacije u 39C7.
To je osnova za Brown/Hurst/fraktal/NIST analizu.
Učitavanje, lex-rank mapiranje, Weierstrass/Brown/Hurst/fraktal aparate, testove i TXT/PNG izlaz.

Skripta radi:
učita loto7_4624_k43.csv
svaku kombinaciju mapira u tačan lex-indeks iz prostora C(39,7)
formira krivu f(t)
računa aparate/testove: Brown inkrementi, Hurst R/S, ACF, mutual information, sample entropy, permutation entropy, Higuchi/Katz fraktalna dimenzija, NIST indikatori
snima 1_KarlWeierstrass.txt
snima 1_KarlWeierstrass.png
Funkcija je u kodu lex_idx (NumPy niz od 4624 vrednosti), f(t) = lex_idx[t-1]. 
Nacrtana je u trenutnom PNG-u kao mali panel gore-levo, ali je tu zbijena. 
Zaseban full-width PNG samo za krivu 1_KarlWeierstrass_v2_1.png (full-width 16×5) sa svih 4624 tačaka.

Kratka analiza prvih rezultata:
Hurst H = 0.593 → blago iznad 0.5, ima slabu perzistenciju (dugoročnu pozitivnu zavisnost), nije čist random walk.
ACF max = 0.029 na lag 21 → ispod praga 0.08, linearna memorija je slaba.
Mutual information lag 1 = 0.023, lag 7 = 0.023 → ispod praga 0.05, ni nelinearno nije jako.
Sample entropy = 2.24, permutation entropy = 0.99997 → suštinski maksimalno haotično.
NIST testovi (monobit, runs, block_freq, cusum, approx_entropy): svi p > 0.01 — ne odbacuju random hipotezu.
Inkrementi (Δf): očekivano anti-perzistentni (ACF lag1 = -0.50) jer differencing samo po sebi ubacuje anti-korelaciju, to nije signal o pravoj strukturi.
Zaključak: jedini blagi indikator je Hurst > 0.5 (perzistencija). 
Ostalo izgleda kao čisto random. 
Ima smisla proveriti da li je taj Hurst stabilan (npr. testirati na pomeranjima prozora i na permutovanoj seriji za poređenje) — inače može da bude artefakt lex-numerisanja.
"""


import csv
import math
import os
import time
from collections import Counter
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


T0 = time.time()

CSV_DRAWS = "/data/loto7_4624_k43.csv"
CSV_ALL_COMBOS = "/data/kombinacije_39C7.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
TXT_PATH = os.path.join(HERE, "1_KarlWeierstrass.txt")
PNG_PATH = os.path.join(HERE, "1_KarlWeierstrass.png")

N_MAX = 39
K_PICK = 7
TOTAL_COMBOS = math.comb(N_MAX, K_PICK)


# ─────────────────────────────────────────────────────────────────────
# 1. Mapiranje izvucene kombinacije u lex-indeks iz prostora 39C7
#    Kombinacije u kombinacije_39C7.csv su lex-sortirane:
#    1,2,3,4,5,6,7  → indeks 1
#    1,2,3,4,5,6,8  → indeks 2
#    ...
#    Zato se isti indeks dobija direktno kombinatornom lex-rank formulom,
#    bez ucitavanja 15.380.937 redova u RAM.
# ─────────────────────────────────────────────────────────────────────
def read_loto_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < K_PICK:
                continue
            try:
                nums = tuple(sorted(int(x) for x in row[:K_PICK]))
            except ValueError:
                continue
            if len(nums) == K_PICK and len(set(nums)) == K_PICK:
                rows.append(nums)
    return rows


def lex_rank_1based(combo, n=N_MAX, k=K_PICK):
    """1-based lexicographic index equivalent to row number in kombinacije_39C7.csv."""
    combo = tuple(sorted(combo))
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo):
        remaining = k - i - 1
        for candidate in range(prev + 1, value):
            rank0 += math.comb(n - candidate, remaining)
        prev = value
    return rank0 + 1


def combo_from_lex_rank_1based(rank, n=N_MAX, k=K_PICK):
    """Inverse lex-rank: indeks iz 1..C(39,7) → loto kombinacija."""
    rank0 = int(rank) - 1
    combo = []
    prev = 0
    for i in range(k):
        remaining = k - i - 1
        for candidate in range(prev + 1, n + 1):
            block = math.comb(n - candidate, remaining)
            if rank0 >= block:
                rank0 -= block
            else:
                combo.append(candidate)
                prev = candidate
                break
    return tuple(combo)


draws = read_loto_csv(CSV_DRAWS)
lex_idx = np.array([lex_rank_1based(row) for row in draws], dtype=np.float64)
t = np.arange(1, len(lex_idx) + 1, dtype=np.float64)

print()
print("KarlWeierstrass / lex-indeks analiza Loto 7/39")
print("=" * 60)
print(f"CSV izvucenih kombinacija: {CSV_DRAWS}")
print(f"CSV prostora 39C7:         {CSV_ALL_COMBOS}")
print(f"Ucitano izvucenja:         {len(draws)}")
print(f"Ukupan prostor C(39,7):    {TOTAL_COMBOS}")
print(f"Prvi lex-indeksi:          {lex_idx[:10].astype(int).tolist()}")
print()


# ─────────────────────────────────────────────────────────────────────
# 2. Aparati: Weierstrass kriva, Brownovo kretanje, Hurst, fraktal
# ─────────────────────────────────────────────────────────────────────
def normalize01(x):
    x = np.asarray(x, dtype=float)
    span = float(x.max() - x.min())
    return (x - x.min()) / (span + 1e-12)


def brown_increments(x):
    return np.diff(np.asarray(x, dtype=float))


def hurst_rs(series, min_window=8, max_window=None):
    """R/S Hurst estimate: slope of log(R/S) vs log(window)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if max_window is None:
        max_window = max(min_window * 2, n // 4)

    windows = []
    w = min_window
    while w <= max_window:
        windows.append(w)
        w = int(w * 1.45) + 1

    rs_values = []
    used_windows = []
    for w in windows:
        chunks = n // w
        if chunks < 2:
            continue
        vals = []
        for i in range(chunks):
            seg = x[i * w:(i + 1) * w]
            seg = seg - seg.mean()
            z = np.cumsum(seg)
            r = z.max() - z.min()
            s = seg.std(ddof=1)
            if s > 0:
                vals.append(r / s)
        if vals:
            used_windows.append(w)
            rs_values.append(float(np.mean(vals)))

    if len(used_windows) < 2:
        return np.nan, [], []
    slope, intercept, r_value, p_value, stderr = stats.linregress(
        np.log(used_windows), np.log(rs_values)
    )
    return float(slope), used_windows, rs_values


def acf_values(series, max_lag=40):
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    vals = []
    for lag in range(1, max_lag + 1):
        num = np.dot(x[:-lag], x[lag:])
        vals.append(float(num / (denom + 1e-12)))
    return np.array(vals)


def mutual_information_lag(series, lag=1, bins=16):
    x = np.asarray(series, dtype=float)
    a = x[:-lag]
    b = x[lag:]
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist / (hist.sum() + 1e-12)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :] + 1e-12)[nz])))


def sample_entropy(series, m=2, r=None, max_points=1200):
    """Sample entropy; uses tail sample to keep runtime reasonable."""
    x = np.asarray(series, dtype=float)
    if len(x) > max_points:
        x = x[-max_points:]
    if r is None:
        r = 0.2 * x.std(ddof=1)
    if r <= 0 or len(x) <= m + 2:
        return np.nan

    def count_matches(mm):
        templates = np.array([x[i:i + mm] for i in range(len(x) - mm + 1)])
        count = 0
        for i in range(len(templates) - 1):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    b = count_matches(m)
    a = count_matches(m + 1)
    if a == 0 or b == 0:
        return np.inf
    return float(-np.log(a / b))


def permutation_entropy(series, order=3, delay=1):
    x = np.asarray(series, dtype=float)
    patterns = []
    limit = len(x) - delay * (order - 1)
    for i in range(limit):
        window = x[i:i + delay * order:delay]
        patterns.append(tuple(np.argsort(window)))
    counts = Counter(patterns)
    probs = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    pe = -np.sum(probs * np.log(probs + 1e-12))
    return float(pe / np.log(math.factorial(order)))


def higuchi_fd(series, kmax=30):
    x = np.asarray(series, dtype=float)
    n = len(x)
    lk = []
    ks = []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            length = np.sum(np.abs(np.diff(x[idx])))
            norm = (n - 1) / (len(idx) * k)
            lm.append(length * norm)
        if lm:
            ks.append(k)
            lk.append(np.mean(lm))
    slope, intercept, r_value, p_value, stderr = stats.linregress(np.log(1 / np.array(ks)), np.log(lk))
    return float(slope)


# Katz FD je dodat zbog kratkog, stabilnog fraktalnog indikatora na jednoj krivi.
def katz_fd(series):
    x = np.asarray(series, dtype=float)
    diffs = np.abs(np.diff(x))
    length = diffs.sum()
    d = np.max(np.abs(x - x[0]))
    n = len(x)
    if length <= 0 or d <= 0:
        return np.nan
    return float(np.log10(n) / (np.log10(d / length) + np.log10(n)))


def nist_bits_from_series(series):
    x = np.asarray(series, dtype=float)
    med = np.median(x)
    return (x > med).astype(int)


def nist_monobit(bits):
    n = len(bits)
    s = np.sum(2 * bits - 1)
    return float(math.erfc(abs(s) / math.sqrt(2 * n)))


def nist_runs(bits):
    n = len(bits)
    pi = bits.mean()
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0
    runs = 1 + np.sum(bits[1:] != bits[:-1])
    return float(math.erfc(abs(runs - 2 * n * pi * (1 - pi)) /
                           (2 * math.sqrt(2 * n) * pi * (1 - pi))))


def nist_block_frequency(bits, block_size=64):
    n_blocks = len(bits) // block_size
    if n_blocks < 2:
        return np.nan
    blocks = bits[:n_blocks * block_size].reshape(n_blocks, block_size)
    pis = blocks.mean(axis=1)
    chi2 = 4 * block_size * np.sum((pis - 0.5) ** 2)
    return float(stats.chi2.sf(chi2, n_blocks))


def nist_cusum(bits):
    steps = 2 * bits - 1
    z = np.max(np.abs(np.cumsum(steps)))
    n = len(bits)
    if z == 0:
        return 1.0
    # Priblizna verzija CUSUM p-value; dovoljna kao indikator u ovoj analizi.
    return float(2 * (1 - stats.norm.cdf(z / math.sqrt(n))))


def nist_approx_entropy(bits, m=2):
    def phi(mm):
        ext = np.concatenate([bits, bits[:mm - 1]])
        counts = Counter(tuple(ext[i:i + mm]) for i in range(len(bits)))
        probs = np.array(list(counts.values()), dtype=float) / len(bits)
        return np.sum(probs * np.log(probs + 1e-12))

    ap_en = phi(m) - phi(m + 1)
    chi2 = 2 * len(bits) * (math.log(2) - ap_en)
    return float(stats.chi2.sf(chi2, 2 ** (m - 1)))


def nist_battery(series):
    bits = nist_bits_from_series(series)
    return {
        "monobit": nist_monobit(bits),
        "runs": nist_runs(bits),
        "block_freq": nist_block_frequency(bits),
        "cusum": nist_cusum(bits),
        "approx_entropy": nist_approx_entropy(bits),
    }


def test_suite(name, series):
    x = np.asarray(series, dtype=float)
    h, _, _ = hurst_rs(x)
    acf = acf_values(x, max_lag=40)
    mi1 = mutual_information_lag(x, lag=1, bins=16)
    mi7 = mutual_information_lag(x, lag=7, bins=16) if len(x) > 7 else np.nan
    se = sample_entropy(x)
    pe = permutation_entropy(x, order=3)
    hfd = higuchi_fd(x, kmax=30)
    kfd = katz_fd(x)
    nist = nist_battery(x)
    return {
        "aparat": name,
        "n": len(x),
        "hurst_rs": h,
        "acf_max_abs_lag1_40": float(np.max(np.abs(acf))),
        "acf_lag_at_max": int(np.argmax(np.abs(acf)) + 1),
        "acf_lag1": float(acf[0]),
        "mutual_info_lag1": mi1,
        "mutual_info_lag7": mi7,
        "sample_entropy": se,
        "permutation_entropy_norm": pe,
        "higuchi_fd": hfd,
        "katz_fd": kfd,
        **{f"nist_{k}": v for k, v in nist.items()},
    }


weier_curve = normalize01(lex_idx)
brown_curve = np.cumsum(brown_increments(weier_curve))
brown_dx = brown_increments(weier_curve)

results = [
    test_suite("Weierstrass lex kriva f(t)", weier_curve),
    test_suite("Brownovo kretanje: kumulativni dx", brown_curve),
    test_suite("Brownovo kretanje: inkrementi dx", brown_dx),
    test_suite("Hurst/R-S nad lex krivom", weier_curve),
    test_suite("Fraktalna dimenzija nad lex krivom", weier_curve),
]


def status_from_results(row):
    signals = []
    if not np.isnan(row["hurst_rs"]):
        if row["hurst_rs"] > 0.57:
            signals.append("H>0.57 perzistentno")
        elif row["hurst_rs"] < 0.43:
            signals.append("H<0.43 anti-perzistentno")
    if row["acf_max_abs_lag1_40"] > 0.08:
        signals.append("ACF pik")
    if row["mutual_info_lag1"] > 0.05 or row["mutual_info_lag7"] > 0.05:
        signals.append("MI signal")
    if row["permutation_entropy_norm"] < 0.95:
        signals.append("niska perm-entropy")
    nist_bad = [
        k for k, v in row.items()
        if k.startswith("nist_") and isinstance(v, float) and not np.isnan(v) and v < 0.01
    ]
    if nist_bad:
        signals.append("NIST p<0.01")
    return ", ".join(signals) if signals else "bez jakog signala"


# ─────────────────────────────────────────────────────────────────────
# 3. Izlaz: TXT + PNG za analizu rezultata pre bilo kakve predikcije
# ─────────────────────────────────────────────────────────────────────
print("Rezultati testova po aparatu:")
print(f"{'Aparat':<42}{'H':>8}{'ACFmax':>10}{'MI1':>10}{'PE':>8}{'HFD':>8}  Signal")
print("-" * 110)
for r in results:
    print(f"{r['aparat']:<42}{r['hurst_rs']:>8.3f}{r['acf_max_abs_lag1_40']:>10.3f}"
          f"{r['mutual_info_lag1']:>10.3f}{r['permutation_entropy_norm']:>8.3f}"
          f"{r['higuchi_fd']:>8.3f}  {status_from_results(r)}")
print()

# Standalone KarlWeierstrass kriva — svih 4624 tacaka, full width radi citljivosti.
KRIVA_PNG = os.path.join(HERE, "1_KarlWeierstrass_kriva.png")
fig_curve, ax_curve = plt.subplots(figsize=(16, 5))
ax_curve.plot(t, lex_idx, linewidth=0.6, color="steelblue")
ax_curve.set_title(
    f"Karlova kriva: f(t) = lex-indeks za svih {len(lex_idx)} izvlacenja "
    f"(opseg 1..{TOTAL_COMBOS:,})",
    fontsize=12,
)
ax_curve.set_xlabel("t (redni broj izvlacenja)")
ax_curve.set_ylabel("lex-indeks u 39C7")
ax_curve.set_xlim(1, len(lex_idx))
ax_curve.grid(True, alpha=0.3)
fig_curve.tight_layout()
fig_curve.savefig(KRIVA_PNG, dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(3, 2, figsize=(14, 11))
fig.suptitle("KarlWeierstrass Loto 7/39 — lex-indeks kriva i testovi strukture",
             fontsize=14, fontweight="bold")

axes[0, 0].plot(t, lex_idx, linewidth=0.8, color="steelblue")
axes[0, 0].set_title("4624 lex-indeksa: f(t)")
axes[0, 0].set_xlabel("redni broj izvlačenja")
axes[0, 0].set_ylabel("lex indeks")

axes[0, 1].plot(t[1:], brown_dx, linewidth=0.8, color="darkorange")
axes[0, 1].set_title("Brown aparat: inkrementi dx = Δf(t)")
axes[0, 1].set_xlabel("redni broj izvlačenja")
axes[0, 1].set_ylabel("normalizovani Δ")

acf = acf_values(weier_curve, max_lag=40)
axes[1, 0].bar(np.arange(1, 41), acf, color="seagreen")
axes[1, 0].axhline(0, color="black", linewidth=0.8)
axes[1, 0].set_title("ACF lex krive (lag 1..40)")
axes[1, 0].set_xlabel("lag")
axes[1, 0].set_ylabel("ACF")

h, hw, hrs = hurst_rs(weier_curve)
axes[1, 1].plot(np.log(hw), np.log(hrs), marker="o", color="purple")
axes[1, 1].set_title(f"Hurst R/S log-log fit (H={h:.3f})")
axes[1, 1].set_xlabel("log(window)")
axes[1, 1].set_ylabel("log(R/S)")

nist_names = ["monobit", "runs", "block_freq", "cusum", "approx_entropy"]
nist_vals = [results[0][f"nist_{name}"] for name in nist_names]
axes[2, 0].bar(nist_names, nist_vals, color="tomato")
axes[2, 0].axhline(0.01, color="black", linestyle="--", linewidth=1, label="p=0.01")
axes[2, 0].set_ylim(0, 1)
axes[2, 0].set_title("NIST indikatori na lex krivi")
axes[2, 0].tick_params(axis="x", rotation=25)
axes[2, 0].legend()

metric_names = ["Hurst", "ACFmax", "MI1", "PermEnt", "HiguchiFD"]
metric_vals = [
    results[0]["hurst_rs"],
    results[0]["acf_max_abs_lag1_40"],
    results[0]["mutual_info_lag1"],
    results[0]["permutation_entropy_norm"],
    results[0]["higuchi_fd"],
]
axes[2, 1].bar(metric_names, metric_vals, color="slateblue")
axes[2, 1].set_title("Sažetak glavnih indikatora (lex kriva)")
axes[2, 1].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
plt.show()

elapsed = time.time() - T0

with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write("KarlWeierstrass algorithm for Loto 7/39 prediction\n")
    f.write("=" * 70 + "\n\n")
    f.write("Ulaz:\n")
    f.write(f"  Izvucene kombinacije: {CSV_DRAWS}\n")
    f.write(f"  Ukupan prostor 39C7:  {CSV_ALL_COMBOS}\n")
    f.write(f"  Ucitano izvucenja:    {len(draws)}\n")
    f.write(f"  C(39,7):              {TOTAL_COMBOS}\n")
    f.write(f"  PNG (multi-panel):    {PNG_PATH}\n")
    f.write(f"  PNG (kriva 4624 tac.):{KRIVA_PNG}\n\n")

    f.write("Mapiranje:\n")
    f.write("  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7 kombinacija\n")
    f.write("  Prvih 10 tacaka krive:\n")
    for i in range(min(10, len(draws))):
        f.write(f"    t={i+1:<4} combo={draws[i]}  lex={int(lex_idx[i])}\n")
    f.write("\n")

    f.write("Rezultati testova po aparatu:\n")
    headers = [
        "aparat", "n", "hurst_rs", "acf_max_abs_lag1_40", "acf_lag_at_max",
        "acf_lag1", "mutual_info_lag1", "mutual_info_lag7",
        "sample_entropy", "permutation_entropy_norm",
        "higuchi_fd", "katz_fd",
        "nist_monobit", "nist_runs", "nist_block_freq",
        "nist_cusum", "nist_approx_entropy",
    ]
    f.write("\t".join(headers + ["signal"]) + "\n")
    for r in results:
        row = []
        for hname in headers:
            val = r[hname]
            if isinstance(val, float):
                row.append(f"{val:.6f}" if np.isfinite(val) else str(val))
            else:
                row.append(str(val))
        row.append(status_from_results(r))
        f.write("\t".join(row) + "\n")

    f.write("\nTumacenje pragova (radno, za izbor narednog modela):\n")
    f.write("  Hurst ~0.5: blizu random walk; >0.57 perzistentno; <0.43 anti-perzistentno.\n")
    f.write("  ACFmax >0.08: moguci linearni memorijski signal u lagovima 1..40.\n")
    f.write("  Mutual information >0.05: moguci nelinearni signal.\n")
    f.write("  Permutation entropy <0.95: redosled nije maksimalno haotican.\n")
    f.write("  NIST p<0.01: taj test odbacuje jednostavnu random hipotezu za binarizovani niz.\n")
    f.write("\nNapomena:\n")
    f.write("  Ovo NIJE predikcija. Ovo je analiza da li uopste postoji struktura za predikciju.\n")
    f.write(f"\nUkupno vreme: {timedelta(seconds=int(elapsed))} ({elapsed:.1f} s)\n")

print(f"PNG saved → {KRIVA_PNG}")
print(f"PNG saved → {PNG_PATH}")
print(f"TXT saved → {TXT_PATH}")
print(f"Ukupno vreme: {timedelta(seconds=int(elapsed))} ({elapsed:.1f} s)")
print()


"""
KarlWeierstrass / lex-indeks analiza Loto 7/39
============================================================
CSV izvucenih kombinacija: /data/loto7_4624_k43.csv
CSV prostora 39C7:         /data/kombinacije_39C7.csv
Ucitano izvucenja:         4624
Ukupan prostor C(39,7):    15380937
Prvi lex-indeksi:          [9774196, 3064646, 14623405, 15178170, 5219568, 10167673, 2903730, 9326447, 1398581, 4751586]

Rezultati testova po aparatu:
Aparat                                           H    ACFmax       MI1      PE     HFD  Signal
--------------------------------------------------------------------------------------------------------------
Weierstrass lex kriva f(t)                   0.593     0.029     0.023   1.000   1.000  H>0.57 perzistentno
Brownovo kretanje: kumulativni dx            0.591     0.029     0.023   1.000   1.000  H>0.57 perzistentno
Brownovo kretanje: inkrementi dx             0.069     0.502     0.180   0.995   1.028  H<0.43 anti-perzistentno, ACF pik, MI signal, NIST p<0.01
Hurst/R-S nad lex krivom                     0.593     0.029     0.023   1.000   1.000  H>0.57 perzistentno
Fraktalna dimenzija nad lex krivom           0.593     0.029     0.023   1.000   1.000  H>0.57 perzistentno

PNG saved → /1_KarlWeierstrass_kriva.png
PNG saved → /1_KarlWeierstrass.png
TXT saved → /1_KarlWeierstrass.txt
Ukupno vreme: 0:00:28 (28.3 s)
"""
