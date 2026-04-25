# India Quant — Multi-Agent Trading System                                                             
                                                                                                 
  Multi-agent quantitative trading system for NSE/BSE equities. Rule-based agents                        
  augmented with optional LLM (OpenRouter free tier) for sentiment and narrative.
                                                                                                         
  ## Quick start                                                                                 
  ```bash                                                                                                
  git clone <this repo>                                                                          
  cd india-quant                                                                                         
  cp .env.example .env  # fill in your keys    
  pip install -r requirements.txt                                                                        
  python main.py --pipeline      # backfill prices                                                       
  python main.py --dashboard     # open http://localhost:5050                                            
                                                                                                         
  See CLAUDE.md for full architecture.                                                                   
                                                                                                 
  3. **Set up branch protection** (Settings → Branches → Add rule on `main`) if you plan to keep working 
  on this. Stops accidental force-pushes.  
