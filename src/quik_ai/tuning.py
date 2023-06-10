class NullScope():
    def __init__(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        pass

class FixedScope():
    def __init__(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        pass

class HyperVariable:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def sample(self, name, hp):
        return None

class HyperBoolean(HyperVariable):
    def sample(self, name, hp):
        return hp.Boolean(name, **self.kwargs)

class HyperChoice(HyperVariable):
    def __init__(self, values, **kwargs):
        super().__init__(**kwargs)
        self.values = values
        
    def sample(self, name, hp):
        return hp.Choice(name, self.values, **self.kwargs)

class HyperFixed(HyperVariable):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        
    def sample(self, name, hp):
        return hp.Fixed(name, self.value, **self.kwargs)

class HyperFloat(HyperVariable):
    def __init__(self, min_value, max_value, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        
    def sample(self, name, hp):
        return hp.Float(name, self.min_value, self.max_value, **self.kwargs)

class HyperInt(HyperVariable):
    def __init__(self, min_value, max_value, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        
    def sample(self, name, hp):
        return hp.Int(name, self.min_value, self.max_value, **self.kwargs)
    
class Tunable:
    def __init__(self, name, **kwargs):
        
        if isinstance(name, (tuple, list)):
            name = ','.join(name)
        
        self.name = name
        
    def _get_hp(self, scope, param_name, hp):
        
        if isinstance(scope, NullScope):
            return None
        
        value = getattr(self, param_name, None)
        
        if value is None or hp is None:
            return value
        
        if not isinstance(value, HyperVariable):
            return value
        
        return value.sample('%s/%s' % (self.name, param_name), hp)
    
    def _apply_hp(self, hp):
        for key, value in hp.items():
            pts = key.split('/')
            
            if len(pts) < 2:
                continue
            
            obj_name = '/'.join(pts[:-1])
            
            if obj_name != self.name:
                continue
            
            if not hasattr(self, pts[-1]):
                continue
            
            setattr(self, pts[-1], value)
    
    def _condition_on_parent(self, hp, parent_param_name, parent_values, scope=None):
        
        if isinstance(scope, NullScope):
            return NullScope()
        
        value = getattr(self, parent_param_name, None)
        
        if isinstance(value, HyperVariable):
            return hp.conditional_scope('%s/%s' % (self.name, parent_param_name), parent_values)
        
        if value not in parent_values:
            return NullScope()
        
        return FixedScope()
    
    def get_parameters(self, hp):
        return {}
    
    def get_dependent_tunables(self):
        return [self]