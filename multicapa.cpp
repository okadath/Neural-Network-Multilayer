#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <utility>
#include <type_traits>
#include <vector>

double logistica(double x)
{
   return 1 / (1 + std::exp(-x));
}

template<typename T, int N>
class matrix {
   struct lista : std::array<int, N> {
      template<typename... P, typename U = typename std::enable_if<sizeof...(P) == N, void>::type>
      lista(const P&... i)
      : std::array<int, N>{{ i... }}
      {
      }
   };
public:
   matrix(const lista& d)
   : dim(d)
   {
      mem.resize(std::accumulate(dim.begin( ), dim.end( ), 1, [](int a, int b) {
         return a * b;
      }));
   }

   int size(int i) const
   {
      return dim[i];
   }

   T& operator[](lista li)
   {
      int indice = 0, producto = 1;
      for (int i = dim.size( ) - 1; i >= 0; producto *= dim[i--]) {
         indice += li[i] * producto;
      }
      return mem[indice];
   }

private:
   std::vector<T> mem;
   std::array<int, N> dim;
};

int main(int argc, const char* argv[])
{
   if (argc <= 1) {
      std::cout << "Uso: " << argv[0] << " archivo\n";
      return 0;
   }

   std::ifstream arch(argv[1]);
   if (!arch.is_open( )) {
      std::cout << "No se pudo abrir " << argv[1] << "\n";
      return 0;
   }

   int e, s, casos_train, casos_test;
   arch >> e >> s >> casos_train >> casos_test;

   std::pair<std::vector<double>, std::vector<double>> instancias_train[casos_train];
   std::pair<std::vector<double>, std::vector<double>> instancias_test[casos_test];
   auto lee_instancias = [&](std::pair<std::vector<double>, std::vector<double>> ins[], int casos) {
      for (int i = 0; i < casos; ++i) {
         std::vector<double> entrada(e), salida(s);
         for (int j = 0; j < e; ++j) {
            arch >> entrada[j];
         }
         for (int j = 0; j < s; ++j) {
            arch >> salida[j];
         }
         ins[i] = { std::move(entrada), std::move(salida) };
      }
   };
   lee_instancias(instancias_train, casos_train);
   lee_instancias(instancias_test, casos_test);

   int capas_ocultas, neuronas_ocultas; double factor; int iteraciones; double error;
   std::cin >> capas_ocultas >> neuronas_ocultas >> factor >> iteraciones >> error;
   int max_neuronas = std::max({ e, s, neuronas_ocultas});

   std::mt19937 gen;
   auto random = [&] {
      return std::uniform_real_distribution<double>(-4 * std::sqrt(6.0 / (e + s)), +4 * std::sqrt(6.0 / (e + s)))(gen);
   };
   matrix<double, 3> w({ capas_ocultas + 2, max_neuronas, max_neuronas });
   matrix<double, 2> u({ capas_ocultas + 2, max_neuronas });
   matrix<double, 2> a({ capas_ocultas + 2, max_neuronas });
   matrix<double, 2> d({ capas_ocultas + 2, max_neuronas });
   std::generate(&w[{ 0, 0, 0 }], &w[{ capas_ocultas + 2, 0, 0 }], random);
   std::generate(&u[{ 0, 0 }],    &u[{ capas_ocultas + 2, 0 }],    random);

   auto neuronas = [&](int c) {
      return (c == 0 ? e : (c == capas_ocultas + 1 ? s : neuronas_ocultas));
   };

   auto evalua_red = [&](const std::vector<double>& entrada) {
      for (int c = 0; c < capas_ocultas + 2; ++c) {
         for (int i = 0; i < neuronas(c); ++i) {
            if (c == 0) {
               a[{ c, i }] = entrada[i];
            } else {
               double temp = u[{ c, i }];
               for (int j = 0; j < neuronas(c - 1); ++j) {
                  temp += w[{ c - 1, j, i }] * a[{ c - 1, j }];
               }
               a[{ c, i }] = logistica(temp);
            }
         }
      }
   };

   auto ajusta_red = [&](const std::vector<double>& esperada) {
      for (int c = capas_ocultas + 1; c > 0; --c) {
         for (int i = 0; i < neuronas(c); ++i) {
            if (c == capas_ocultas + 1) {
               d[{ c, i }] = a[{ c, i }] * (1 - a[{ c, i }]) * (esperada[i] - a[{ capas_ocultas + 1, i }]);
            } else {
               double temp = 0;
               for (int j = 0; j < neuronas(c + 1); ++j) {
                  temp += w[{ c, i, j }] * d[{ c + 1, j }];
               }
               d[{ c, i }] = a[{ c, i }] * (1 - a[{ c, i }]) * temp;
            }
         }
      }

      for (int c = 1; c < capas_ocultas + 2; ++c) {
         for (int i = 0; i < neuronas(c); ++i) {
            u[{ c, i }] += factor * d[{ c, i }];
            for (int j = 0; j < neuronas(c - 1); ++j) {
               w[{ c - 1, j, i}] += factor * d[{ c, i }] * a[{ c - 1, j }];
            }
         }
      }
   };

   auto calcula_error = [&](const std::vector<double>& esperada) {
      double error = 0;
      for (int i = 0; i < neuronas(capas_ocultas + 1); ++i) {
         error += std::pow(esperada[i] - a[{ capas_ocultas + 1, i }], 2);
      }
      return error / 2;
   };

   auto mejor_w = w;
   auto mejor_u = u;
   double mejor_error = std::numeric_limits<double>::max( );
   for (int t = 0; t < iteraciones && error < mejor_error; ++t) {
      std::shuffle(instancias_train, instancias_train + casos_train, gen);
      for (int i = 0; i < casos_train; ++i) {
         evalua_red(instancias_train[i].first);
         ajusta_red(instancias_train[i].second);
      }

      double error_actual = 0;
      for (int i = 0; i < casos_train; ++i) {
         evalua_red(instancias_train[i].first);
         error_actual += calcula_error(instancias_train[i].second);
      }
      error_actual /= casos_train;

      if (error_actual < mejor_error) {
         mejor_w = w;
         mejor_u = u;
      }
   }

   w = mejor_w;
   u = mejor_u;
   double error_tests = 0;
   for (int i = 0; i < casos_test; ++i) {
      evalua_red(instancias_test[i].first);
      error_tests += calcula_error(instancias_test[i].second);
      for (int j = 0; j < instancias_test[i].second.size( ); ++j) {
         std::cout << "{ ye: " << instancias_test[i].second[j] << ", yo: " << a[{ capas_ocultas + 1, j }] << "} ";
      }
      std::cout << "\n";
   }
   std::cout << error_tests / casos_test << "\n";
}
