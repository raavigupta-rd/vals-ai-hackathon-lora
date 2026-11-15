#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;

    // Double rolling hash, O(N) memory and O(N^2 log N) time by scanning one length at a time
    static constexpr uint64_t M1 = 1'000'000'007ULL, M2 = 1'000'000'009ULL;
    static constexpr uint64_t B1 = 911382323ULL, B2 = 972663749ULL;
    vector<uint64_t> p1(n + 1, 1), p2(n + 1, 1), h1(n + 1, 0), h2(n + 1, 0);
    for (int i = 0; i < n; i++) {
        p1[i + 1] = p1[i] * B1 % M1;
        p2[i + 1] = p2[i] * B2 % M2;
        h1[i + 1] = (h1[i] * B1 + (unsigned char)s[i]) % M1;
        h2[i + 1] = (h2[i] * B2 + (unsigned char)s[i]) % M2;
    }

    vector<long long> by_freq(n + 1, 0);
    vector<uint64_t> cur;
    cur.reserve(n);
    for (int len = 1; len <= n; len++) {
        cur.clear();
        for (int i = 0; i + len <= n; i++) {
            uint64_t x1 = (h1[i + len] + M1 - h1[i] * p1[len] % M1) % M1;
            uint64_t x2 = (h2[i + len] + M2 - h2[i] * p2[len] % M2) % M2;
            cur.push_back((x1 << 32) | x2);
        }
        sort(cur.begin(), cur.end());
        for (size_t i = 0; i < cur.size(); ) {
            size_t j = i + 1;
            while (j < cur.size() && cur[j] == cur[i]) j++;
            by_freq[j - i]++;
            i = j;
        }
    }

    vector<long long> pref(n + 1, 0);
    for (int f = 1; f <= n; f++) pref[f] = pref[f - 1] + by_freq[f];
    while (q--) {
        int a, b;
        cin >> a >> b;
        cout << (pref[b] - pref[a - 1]) << '\n';
    }
    return 0;
}
