#include <bits/stdc++.h>
using namespace std;

struct BinaryTrie {
    vector<array<int, 2>> ch;
    int sz = 0;
    void init(int max_nodes) {
        ch.resize(max_nodes);
        reset();
    }
    void reset() {
        ch[0] = {-1, -1};
        sz = 1;
    }
    void insert(int x, int top_bit) {
        int u = 0;
        for (int b = top_bit; b >= 0; b--) {
            int bit = (x >> b) & 1;
            if (ch[u][bit] == -1) {
                ch[sz] = {-1, -1};
                ch[u][bit] = sz++;
            }
            u = ch[u][bit];
        }
    }
    int min_xor(int x, int top_bit) const {
        int u = 0, res = 0;
        for (int b = top_bit; b >= 0; b--) {
            int bit = (x >> b) & 1;
            if (ch[u][bit] != -1) {
                u = ch[u][bit];
            } else {
                res |= (1 << b);
                u = ch[u][bit ^ 1];
            }
        }
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    sort(a.begin(), a.end());
    a.erase(unique(a.begin(), a.end()), a.end());
    int m = (int)a.size();
    if (m <= 1) {
        cout << 0 << '\n';
        return 0;
    }

    BinaryTrie trie;
    trie.init(m * 30 + 5);

    function<long long(int, int, int)> solve = [&](int l, int r, int bit) -> long long {
        if (bit < 0 || l >= r) return 0;
        int mid = l;
        while (mid <= r && !((a[mid] >> bit) & 1)) mid++;
        long long ans = solve(l, mid - 1, bit - 1) + solve(mid, r, bit - 1);
        if (l < mid && mid <= r) {
            trie.reset();
            int L1 = l, R1 = mid - 1, L2 = mid, R2 = r;
            if (R1 - L1 > R2 - L2) { swap(L1, L2); swap(R1, R2); }
            for (int i = L1; i <= R1; i++) trie.insert(a[i], bit - 1);
            int best = INT_MAX;
            for (int i = L2; i <= R2; i++) best = min(best, trie.min_xor(a[i], bit - 1));
            ans += (1LL << bit) + best;
        }
        return ans;
    };

    cout << solve(0, m - 1, 29) << '\n';
    return 0;
}
