#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> A(n + 1), W(n + 1), F(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> A[i];
    for (int i = 1; i <= n; i++) cin >> W[i];
    F[0] = 1;
    for (int i = 1; i <= n; i++) {
        unsigned __int128 acc = 0;
        for (int j = 0; j < i; j++) acc += (unsigned __int128)F[j] * W[i - j];
        F[i] = (long long)((acc % MOD) * A[i] % MOD);
    }
    for (int i = 1; i <= n; i++) cout << F[i] << (i == n ? '\n' : ' ');
    return 0;
}
