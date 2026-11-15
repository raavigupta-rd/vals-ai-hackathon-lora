#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int H, W, K;
    cin >> H >> W >> K;
    vector<vector<char>> bad(H + 1, vector<char>(W + 1, 0));
    for (int i = 0; i < K; i++) {
        int r, c; cin >> r >> c;
        bad[r][c] = 1;
    }
    vector<long long> dp(W + 1, 0);
    dp[1] = 1;
    for (int r = 1; r <= H; r++) {
        for (int c = 1; c <= W; c++) {
            if (bad[r][c]) dp[c] = 0;
            else if (!(r == 1 && c == 1)) dp[c] = (dp[c] + dp[c - 1]) % MOD;
        }
    }
    cout << dp[W] << '\n';
    return 0;
}
