#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<pair<int, int>> iv(n);
    for (int i = 0; i < n; i++) cin >> iv[i].second >> iv[i].first; // (R, L)
    sort(iv.begin(), iv.end());

    multiset<int> ends;
    int ans = 0;
    for (auto [r, l] : iv) {
        auto it = ends.upper_bound(l);
        if (it != ends.begin()) {
            --it;
            ends.erase(it);
            ends.insert(r);
            ans++;
        } else if ((int)ends.size() < k) {
            ends.insert(r);
            ans++;
        }
    }
    cout << ans << '\n';
    return 0;
}
