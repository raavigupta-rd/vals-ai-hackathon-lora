#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;
const unsigned long long MOD2 = 16ULL * MOD * MOD;
using Mat = array<array<long long, 65>, 65>;

Mat mul(const Mat& A, const Mat& B, int n) {
    Mat C{};
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            unsigned long long s = 0;
            for (int k = 0; k < n; k++) {
                s += (unsigned long long)A[i][k] * B[k][j];
                if (s >= MOD2) s -= MOD2;
            }
            C[i][j] = (long long)(s % MOD);
        }
    }
    return C;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    const int LOG = 60;
    vector<Mat> pw(LOG);
    for (auto& M : pw) for (auto& row : M) row.fill(0);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        pw[0][u - 1][v - 1] = (pw[0][u - 1][v - 1] + 1) % MOD;
    }
    for (int b = 1; b < LOG; b++) pw[b] = mul(pw[b - 1], pw[b - 1], n);

    array<long long, 65> vec, nvec;
    while (q--) {
        int u, v;
        long long k;
        cin >> u >> v >> k;
        vec.fill(0);
        vec[u - 1] = 1;
        for (int b = 0; b < LOG; b++) if ((k >> b) & 1LL) {
            const Mat& M = pw[b];
            for (int j = 0; j < n; j++) {
                unsigned long long s = 0;
                for (int x = 0; x < n; x++) {
                    s += (unsigned long long)vec[x] * M[x][j];
                    if (s >= MOD2) s -= MOD2;
                }
                nvec[j] = (long long)(s % MOD);
            }
            vec = nvec;
        }
        cout << vec[v - 1] << '\n';
    }
    return 0;
}
