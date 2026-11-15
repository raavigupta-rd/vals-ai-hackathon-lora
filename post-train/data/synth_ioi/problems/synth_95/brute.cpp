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

// 40-pt subtask solver: LGV Lemma + K rolling-row O(H * W) grid DPs (O(W) RAM, cleanly TLEs on large)
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int H, W, K, B;
    cin >> H >> W >> K >> B;
    vector<int> S(K), T(K);
    for (int i = 0; i < K; i++) cin >> S[i];
    for (int i = 0; i < K; i++) cin >> T[i];
    vector<vector<int>> blocked_in_row(H + 1);
    for (int i = 0; i < B; i++) {
        int r, c; cin >> r >> c;
        blocked_in_row[r].push_back(c);
    }

    vector<vector<long long>> M(K, vector<long long>(K, 0));
    vector<long long> dp(W + 1, 0);
    vector<char> bad(W + 1, 0);

    for (int i = 0; i < K; i++) {
        fill(dp.begin(), dp.end(), 0);
        dp[S[i]] = 1;
        for (int r = 1; r <= H; r++) {
            for (int c : blocked_in_row[r]) bad[c] = 1;
            for (int c = 1; c <= W; c++) {
                if (bad[c]) {
                    dp[c] = 0;
                } else if (r == 1 && c == S[i]) {
                    dp[c] = 1;
                } else {
                    dp[c] += dp[c - 1];
                    if (dp[c] >= MOD) dp[c] -= MOD;
                }
            }
            for (int c : blocked_in_row[r]) bad[c] = 0;
        }
        for (int j = 0; j < K; j++) M[i][j] = dp[T[j]];
    }

    long long det = 1;
    for (int col = 0; col < K; col++) {
        int piv = -1;
        for (int row = col; row < K; row++) if (M[row][col]) { piv = row; break; }
        if (piv == -1) { det = 0; break; }
        if (piv != col) { swap(M[piv], M[col]); det = (MOD - det) % MOD; }
        det = det * M[col][col] % MOD;
        long long inv_p = modpow(M[col][col], MOD - 2);
        for (int row = col + 1; row < K; row++) if (M[row][col]) {
            long long f = M[row][col] * inv_p % MOD;
            for (int c = col; c < K; c++) {
                M[row][c] = (M[row][c] - f * M[col][c]) % MOD;
                if (M[row][c] < 0) M[row][c] += MOD;
            }
        }
    }
    cout << det << '\n';
    return 0;
}
