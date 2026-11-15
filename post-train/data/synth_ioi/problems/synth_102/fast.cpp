#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int b, q;
    if (!(cin >> b >> q)) return 0;
    int N = 1 << b;
    vector<vector<int>> fa(b + 1, vector<int>(N, 0));
    vector<vector<int>> fb(b + 1, vector<int>(N, 0));
    for (int m = 0; m < N; m++) {
        int x;
        cin >> x;
        fa[__builtin_popcount((unsigned)m)][m] = x % MOD;
    }
    for (int m = 0; m < N; m++) {
        int x;
        cin >> x;
        fb[__builtin_popcount((unsigned)m)][m] = x % MOD;
    }

    for (int r = 0; r <= b; r++) {
        auto& A = fa[r];
        auto& B = fb[r];
        for (int i = 0; i < b; i++) {
            int bit = 1 << i;
            for (int m = 0; m < N; m++) {
                if (m & bit) {
                    int u = A[m] + A[m ^ bit];
                    A[m] = (u >= MOD ? u - MOD : u);
                    int v = B[m] + B[m ^ bit];
                    B[m] = (v >= MOD ? v - MOD : v);
                }
            }
        }
    }

    vector<vector<int>> fc(b + 1, vector<int>(N, 0));
    for (int r = 0; r <= b; r++) {
        auto& C = fc[r];
        for (int j = 0; j <= r; j++) {
            const auto& A = fa[j];
            const auto& B = fb[r - j];
            for (int m = 0; m < N; m++) {
                C[m] = (int)((C[m] + 1LL * A[m] * B[m]) % MOD);
            }
        }
        for (int i = 0; i < b; i++) {
            int bit = 1 << i;
            for (int m = 0; m < N; m++) {
                if (m & bit) {
                    int u = C[m] - C[m ^ bit];
                    C[m] = (u < 0 ? u + MOD : u);
                }
            }
        }
    }

    while (q--) {
        int s;
        cin >> s;
        cout << fc[__builtin_popcount((unsigned)s)][s] << '\n';
    }
    return 0;
}
