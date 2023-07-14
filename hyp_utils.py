import torch
# Hyperbolic addition
def hyp_add(x, y, c):
    num_x = (1+ 2*c*torch.matmul(y,torch.unsqueeze(x,dim=1))+ torch.unsqueeze((c*torch.norm(y,dim=1)**2),dim=1))
    num_y = (1 - c * torch.norm(x)**2)
    num = num_x*x + num_y* y
    denom = torch.squeeze(1/(1 + 2 * c * torch.matmul(y,torch.unsqueeze(x,dim=1)) + c**2 * torch.norm(x)**2 * torch.unsqueeze(torch.norm(y, dim=1)**2,dim=1)),dim=1)
    return num * denom[:, None]

# Apply Exponential Map to Hyperpolic 
def exp_map(x,v, c):
    
    # Calculate lam_c using vectorized operations
    lam_c = 2 / (1 - c * torch.norm(x) ** 2)

    # Calculate norms of all rows in v using torch.norm and keep them in a separate tensor
    v_norms = torch.norm(v, dim=1)

    # Calculate a for all rows in v using vectorized operations
    a = (c ** 0.5) * lam_c * v_norms / 2
    a = (torch.tanh(a)[:,None] * v) / ((c ** 0.5) * v_norms[:, None])        

    return hyp_add(x, a, c)

#Hyperbolic distance
def d_hyp(x,y,c):
    return torch.arctanh(c**(0.5)*torch.norm(-hyp_add(x,y,c),dim=1))*2/((c)**(0.5))