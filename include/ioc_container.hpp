#include <unordered_map>
#include <typeindex>
#include <memory>
#include <typeinfo>

namespace IoC {

class DiItem {
  std::type_info const* type_;
  std::shared_ptr<void> item_;

 public:
  DiItem() : type_(&typeid(nullptr)) {}
  DiItem(std::nullptr_t) : DiItem() {}

  template<class T>
  DiItem(std::shared_ptr<T> item) : type_(&typeid(T)), item_(item) {}

  template<class T>
  std::shared_ptr<T> get() {
    return typeid(T) == *type_ ? std::static_pointer_cast<T>(item_) : nullptr;
  }
};

class Container {
  using FactoryFn = std::function<DiItem(Container& resolver)>;
  typedef std::unordered_map<std::type_index, FactoryFn> ItemsMapType;
  ItemsMapType items_;

 public:
  template<class T>
  std::shared_ptr<T> Resolve() {
    auto it = items_.find(typeid(T));
    return it == items_.end() ? nullptr : it->second(*this).get<T>();
  }

  template<class T, class I, class ...Args>
  FactoryFn CreateFactory() {
    return [=](Container& resolver) mutable {
      std::shared_ptr<I> i = std::make_shared<T>(resolver.Resolve<Args>()...);
      return DiItem(i);
    };
  }

  template<class T, class I, class ...Args>
  void RegisterType() {
    items_.erase(std::type_index(typeid(I)));
    items_.insert(make_pair(std::type_index(typeid(I)), CreateFactory<T, I, Args...>()));
  }

  template<class T, class I, class ...Args>
  void RegisterInstance() {
    items_.erase(std::type_index(typeid(I)));
    auto is_created = false;
    DiItem singleton;
    auto singleton_factory = [=](Container& resolver) mutable {
      if (is_created) return singleton;
      is_created = true;
      return singleton = CreateFactory<T, I, Args...>()(resolver);
    };
    items_.insert(make_pair(std::type_index(typeid(I)), singleton_factory));
  }

  static Container& Get() {
    static auto container = std::make_shared<Container>();
    return *container;
  }
};

} // End of IoC namespace.
