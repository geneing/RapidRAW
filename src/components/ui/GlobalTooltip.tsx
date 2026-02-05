import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { AnimatePresence, motion } from 'framer-motion';
import clsx from 'clsx';

const TOOLTIP_DELAY = 500;
const OFFSET = 8;

export default function GlobalTooltip() {
  const [isVisible, setIsVisible] = useState(false);
  const [content, setContent] = useState('');
  const [position, setPosition] = useState({ x: 0, y: 0 });
  
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const currentTargetRef = useRef<HTMLElement | null>(null);

  const clearPendingTooltip = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const restoreTitle = (element: HTMLElement | null) => {
    if (element) {
      const originalTitle = element.getAttribute('data-custom-tooltip');
      if (originalTitle) {
        element.setAttribute('title', originalTitle);
        element.removeAttribute('data-custom-tooltip');
      }
    }
  };

  const hideTooltip = () => {
    clearPendingTooltip();
    setIsVisible(false);
    restoreTitle(currentTargetRef.current);
    currentTargetRef.current = null;
  };

  useEffect(() => {
    const handleMouseOver = (e: MouseEvent) => {
      const target = (e.target as HTMLElement).closest('[title], [data-custom-tooltip]');
      
      if (!target || !(target instanceof HTMLElement)) {
        return;
      }

      if (target === currentTargetRef.current) {
        return;
      }

      const titleText = target.getAttribute('title') || target.getAttribute('data-custom-tooltip');
      
      if (!titleText) {
        return;
      }

      if (currentTargetRef.current && currentTargetRef.current !== target) {
        hideTooltip();
      }

      if (target.hasAttribute('title')) {
        target.setAttribute('data-custom-tooltip', titleText);
        target.removeAttribute('title');
      }
      
      currentTargetRef.current = target;

      clearPendingTooltip();

      timerRef.current = setTimeout(() => {
        const rect = target.getBoundingClientRect();

        let x = rect.left + rect.width / 2;
        let y = rect.bottom + OFFSET;

        const viewportWidth = window.innerWidth;
        x = Math.max(20, Math.min(x, viewportWidth - 20));

        if (y + 40 > window.innerHeight) {
          y = rect.top - OFFSET; 
        }

        setContent(titleText);
        setPosition({ x, y });
        setIsVisible(true);
      }, TOOLTIP_DELAY);
    };

    const handleMouseOut = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const relatedTarget = e.relatedTarget as HTMLElement | null;

      if (!currentTargetRef.current) return;
      
      const isLeavingCurrentTarget = 
        target === currentTargetRef.current || 
        currentTargetRef.current.contains(target);

      const isMovingToChild = 
        relatedTarget && 
        currentTargetRef.current.contains(relatedTarget);

      if (isLeavingCurrentTarget && !isMovingToChild) {
        hideTooltip();
      }
    };

    const handleMouseDown = () => {
      hideTooltip();
    };

    const handleScroll = () => {
      hideTooltip();
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        hideTooltip();
      }
    };

    document.addEventListener('mouseover', handleMouseOver);
    document.addEventListener('mouseout', handleMouseOut);
    document.addEventListener('mousedown', handleMouseDown);
    document.addEventListener('scroll', handleScroll, true);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mouseover', handleMouseOver);
      document.removeEventListener('mouseout', handleMouseOut);
      document.removeEventListener('mousedown', handleMouseDown);
      document.removeEventListener('scroll', handleScroll, true);
      document.removeEventListener('keydown', handleKeyDown);
      clearPendingTooltip();
    };
  }, []);

  const isAbove = currentTargetRef.current 
    ? position.y < currentTargetRef.current.getBoundingClientRect().top 
    : false;

  return createPortal(
    <AnimatePresence mode="wait">
      {isVisible && (
        <motion.div
          key={content}
          initial={{ opacity: 0, scale: 0.9, y: isAbove ? 5 : -5, x: "-50%" }}
          animate={{ opacity: 1, scale: 1, y: isAbove ? -10 : 0, x: "-50%" }}
          exit={{ opacity: 0, scale: 0.9, x: "-50%" }}
          transition={{ duration: 0.15, ease: "easeOut" }}
          style={{ 
            top: position.y, 
            left: position.x,
          }}
          className={clsx(
            "fixed z-[100] pointer-events-none",
            "bg-surface/80 backdrop-blur-sm text-text-primary",
            "border border-text-secondary/10 shadow-xl rounded-md",
            "px-2.5 py-1.5 text-xs font-medium whitespace-nowrap",
            isAbove && "-translate-y-full" 
          )}
        >
          {content}
        </motion.div>
      )}
    </AnimatePresence>,
    document.body
  );
}