#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<pair<int, int>> iv(n);
    for (int i = 0; i < n; i++) cin >> iv[i].second >> iv[i].first;
    sort(iv.begin(), iv.end());

    vector<int> ends(k, 0);
    int ans = 0;
    for (auto [r, l] : iv) {
        int best_idx = -1;
        for (int j = 0; j < k; j++) {
            if (ends[j] <= l && (best_idx == -1 || ends[j] > ends[best_idx])) {
                best_idx = j;
            }
        }
        if (best_idx != -1) {
            ends[best_idx] = r;
            ans++;
        }
    }
    cout << ans << '\n';
    return 0;
}
