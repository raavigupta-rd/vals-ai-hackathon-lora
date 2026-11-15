#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int H, W, K, B;
    cin >> H >> W >> K >> B;
    vector<int> S(K), T(K);
    for (int i = 0; i < K; i++) cin >> S[i];
    for (int i = 0; i < K; i++) cin >> T[i];

    vector<pair<int, int>> obs(B);
    for (int i = 0; i < B; i++) cin >> obs[i].first >> obs[i].second;
    sort(obs.begin(), obs.end());

    int MAXF = H + W + 5;
    vector<long long> fac(MAXF, 1), ifac(MAXF, 1);
    for (int i = 1; i < MAXF; i++) fac[i] = fac[i - 1] * i % MOD;
    ifac[MAXF - 1] = modpow(fac[MAXF - 1], MOD - 2);
    for (int i = MAXF - 2; i >= 0; i--) ifac[i] = ifac[i + 1] * (i + 1) % MOD;

    auto ways = [&](int r1, int c1, int r2, int c2) -> long long {
        int dr = r2 - r1, dc = c2 - c1;
        if (dr < 0 || dc < 0) return 0;
        return fac[dr + dc] * ifac[dr] % MOD * ifac[dc] % MOD;
    };

    // Precompute obstacle-to-obstacle free path counts
    vector<vector<long long>> oo(B, vector<long long>(B, 0));
    for (int a = 0; a < B; a++) {
        for (int b = a + 1; b < B; b++) {
            oo[a][b] = ways(obs[a].first, obs[a].second, obs[b].first, obs[b].second);
        }
    }

    // Build K x K LGV path matrix M
    vector<vector<long long>> M(K, vector<long long>(K, 0));
    vector<long long> dp_obs(B);

    for (int i = 0; i < K; i++) {
        // dp_obs[a] = paths from (1, S[i]) to obs[a] avoiding all earlier obstacles
        for (int a = 0; a < B; a++) {
            long long val = ways(1, S[i], obs[a].first, obs[a].second);
            for (int p = 0; p < a; p++) if (dp_obs[p] && oo[p][a]) {
                val = (val - dp_obs[p] * oo[p][a]) % MOD;
            }
            if (val < 0) val += MOD;
            dp_obs[a] = val;
        }
        for (int j = 0; j < K; j++) {
            long long val = ways(1, S[i], H, T[j]);
            for (int a = 0; a < B; a++) if (dp_obs[a]) {
                val = (val - dp_obs[a] * ways(obs[a].first, obs[a].second, H, T[j])) % MOD;
            }
            if (val < 0) val += MOD;
            M[i][j] = val;
        }
    }

    // Determinant of M mod 998244353
    long long det = 1;
    for (int col = 0; col < K; col++) {
        int piv = -1;
        for (int row = col; row < K; row++) if (M[row][col] != 0) { piv = row; break; }
        if (piv == -1) { det = 0; break; }
        if (piv != col) {
            swap(M[piv], M[col]);
            det = (MOD - det) % MOD;
        }
        det = det * M[col][col] % MOD;
        long long inv_p = modpow(M[col][col], MOD - 2);
        for (int row = col + 1; row < K; row++) if (M[row][col] != 0) {
            long long factor = M[row][col] * inv_p % MOD;
            for (int c = col; c < K; c++) {
                M[row][c] = (M[row][c] - factor * M[col][c]) % MOD;
                if (M[row][c] < 0) M[row][c] += MOD;
            }
        }
    }

    cout << det << '\n';
    return 0;
}
