#include <bits/stdc++.h>
using namespace std;

using ll = long long;
template<typename T> using vec = vector<T>;
template<typename T> using deq = deque<T>;
template<typename T> using p = pair<T, T>;

#define yccc ios_base::sync_with_stdio(false), cin.tie(0)
#define endl '\n'
#define al(a) a.begin(),a.end()
#define F first
#define S second
#define eb emplace_back

int main() {
	yccc;

	int width = 10, height = 6;
	vec<int> _list((height + 2) * (width + 2));

	for (int i = 0; i < height; i++) {
		for (int k = 0; k < width; k++) {
			_list[(i + 1) * (width + 2) + k + 1] = i * width + k;
		}
	}

	for (int i = 0; i < height + 2; i++) {
		for (int k = 0; k < width + 2; k++) {
			cout << _list[i * (width + 2) + k] << "\t";
		}
		cout << endl;
	}
}