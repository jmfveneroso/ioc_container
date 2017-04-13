#include <iostream>
#include "ioc_container.hpp"

class A {
  int x = 0;
 public:
  void HelloWorld() {
    ++x;
    std::cout << "Hello World! " << x << std::endl;
  }
};

class C {
 public:
  virtual void CallA() = 0;
};

class B : public C {
  std::shared_ptr<A> a_;

 public:
  B(std::shared_ptr<A> a) : a_(a) {}

  void CallA() override {
    a_->HelloWorld();
  }

};

class D : public C {
 public:
  D() {}

  void CallA() override {
    std::cout << "This implementation overrides the previous one." << std::endl;
  }

};

class E {
  int x_ = 0;

 public:
  E() {}

  void Call() {
    std::cout << "x: " << x_++ << std::endl;
  }

};

class ContainerBootstrapper {
 public:
  static void Bootstrap() {
    static IoC::Container& container = IoC::Container::Get();
    container.RegisterInstance<E, E>();
    container.RegisterInstance<B, C, A>();
    container.RegisterType<A, A>();
  }
};

int main () {
  ContainerBootstrapper::Bootstrap();

  static IoC::Container& container = IoC::Container::Get();
  std::shared_ptr<C> b = container.Resolve<C>();
  b->CallA();
  b->CallA();

  std::shared_ptr<C> c = container.Resolve<C>();
  c->CallA();
  c->CallA();
  c->CallA();
  b->CallA();

  container.RegisterInstance<D, C>();
  std::shared_ptr<C> d = container.Resolve<C>();
  d->CallA();
  b->CallA();
  c->CallA();

  std::shared_ptr<E> e = container.Resolve<E>();
  e->Call();
  e->Call();
  e->Call();

  return 0;
}
