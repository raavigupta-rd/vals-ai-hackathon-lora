#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 1'000'000'007;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> a(n), len(n, 1), cnt(n, 1);
    for (int i = 0; i < n; i++) cin >> a[i];
    int best_len = 0, best_cnt = 0;
    for (int i = 0; i < n; i++) {
        int l = 1, c = 1;
        for (int j = 0; j < i; j++) {
            if (a[j] < a[i]) {
                if (len[j] + 1 > l) {
                    l = len[j] + 1;
                    c = cnt[j];
                } else if (len[j] + 1 == l) {
                    c += cnt[j];
                    if (c >= MOD) c -= MOD;
                }
            }
        }
        len[i] = l;
        cnt[i] = c;
        if (l > best_len) {
            best_len = l;
            best_cnt = c;
        } else if (l == best_len) {
            best_cnt += c;
            if (best_cnt >= MOD) best_cnt -= MOD;
        }
    }
    cout << best_len << ' ' << best_cnt << '\n';
    return 0;
}
