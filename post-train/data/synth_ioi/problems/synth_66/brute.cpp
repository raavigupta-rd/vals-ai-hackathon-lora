#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    unordered_map<int, long long> cnt;
    cnt.reserve(4096);
    for (int i = 0; i < n; i++) {
        int g = 0;
        for (int j = i; j < n; j++) {
            g = std::gcd(g, a[j]);
            cnt[g]++;
        }
    }
    while (q--) {
        int x; cin >> x;
        auto it = cnt.find(x);
        cout << (it == cnt.end() ? 0LL : it->second) << '\n';
    }
    return 0;
}
